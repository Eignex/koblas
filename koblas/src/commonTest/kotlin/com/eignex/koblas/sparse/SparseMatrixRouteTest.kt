package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.PanelWork
import com.eignex.koblas.vendor.RouteKind
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * What a sparse matrix call reports about itself.
 *
 * The property under test is that the report names the code that runs rather than the engine that was asked
 * for. A whole sparse call is portable CSC scheduling on every engine, so an arm whose Level 1 kernels are
 * Vector API ones must not come out labelled as a vectorised sparse product, and a call whose units of work
 * reach two different kernels must not come out labelled as either one.
 */
class SparseMatrixRouteTest {

    private companion object {
        /** Long enough to clear any indexed crossover a host has, and small enough to build in a test. */
        const val LONG_COLUMN = 4096
    }

    private fun uniform(columns: Int, perColumn: Int): SparseMatrix = SparseMatrix.ofColumns(
        perColumn,
        columns,
        List(columns) { List(perColumn) { row -> row to (row + 1.0) } },
    )

    @Test
    fun `every sparse matrix operation reports the portable scheduling that owns it`() {
        val a = uniform(columns = 4, perColumn = 32)
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            for (operation in SparseMatrixOperation.entries) {
                val route = engine.routeOf(
                    operation,
                    SparseCall(a, destinationElements = 512, depth = 32, updateRun = 512),
                )
                assertEquals("portable-csc", route.scheduling, "${engine.name} $operation")
                assertEquals("portable-csc", engine.sparseImplementation, engine.name)
                assertTrue(
                    route.implementation.startsWith("portable-csc"),
                    "${engine.name} $operation claimed ${route.implementation}",
                )
            }
        }
    }

    @Test
    fun `an operation whose arithmetic is its own traversal names no component`() {
        val route = BuiltinEngines.scalar.routeOf(
            SparseMatrixOperation.GemmSparse,
            SparseCall(uniform(columns = 2, perColumn = 8)),
        )

        assertEquals(emptyList(), route.components)
        assertEquals("portable-csc", route.implementation)
        assertEquals(RouteKind.Direct, route.kind)
        assertTrue(route.exactlyMeasurable)
    }

    @Test
    fun `a destination multiplier that scales names the dense kernel that scales it`() {
        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar
        val a = uniform(columns = 2, perColumn = 8)

        val scaled = engine.routeOf(
            SparseMatrixOperation.Symv,
            SparseCall(a, alpha = 1.0, beta = 0.5, destinationElements = 4096),
        )
        val overwritten = engine.routeOf(
            SparseMatrixOperation.Symv,
            SparseCall(a, alpha = 1.0, beta = 0.0, destinationElements = 4096),
        )

        val expected = assertNotNull(engine.routeOf(DenseOperation.Scale, 4096).implementation)
        assertEquals(listOf("$expected/scale"), scaled.components, "a non-unit beta scales the destination")
        assertEquals(emptyList(), overwritten.components, "a zero beta fills rather than scaling")
    }

    @Test
    fun `a scattered gemv names the indexed leaf its columns reach`() {
        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar
        val short = engine.routeOf(SparseMatrixOperation.Gemv, SparseCall(uniform(4, 1)))
        val long = engine.routeOf(SparseMatrixOperation.Gemv, SparseCall(uniform(4, 4096)))

        for (route in listOf(short, long)) {
            val component = route.components.single()
            assertTrue(component.endsWith("/axpy"), "gemv reported $component")
            assertEquals("portable-csc+$component", route.implementation)
            assertEquals(RouteKind.Direct, route.kind)
        }
        assertEquals(
            engine.sparseKernels.routeOf(SparseOperation.Axpy, 4096).implementation,
            long.components.single().substringBefore('/'),
            "the sparse gemv leaf must be the one a Level 1 axpy of that width reaches",
        )
    }

    /**
     * Columns of one matrix need not be the same length, so they need not reach the same kernel. Naming one of
     * them would publish the other's time under its label, which is the thing this reporting exists to stop.
     *
     * Which lengths straddle a crossover is the host's answer, not this test's: a host whose indexed stores
     * the Vector API cannot use runs the scalar kernel at every length, and there is then no mixture to find.
     */
    @Test
    fun `columns that straddle the indexed crossover are reported as a composition`() {
        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar
        val kernels = engine.sparseKernels
        val short = kernels.routeOf(SparseOperation.Axpy, 1).implementation
        val long = kernels.routeOf(SparseOperation.Axpy, LONG_COLUMN).implementation
        if (short == long) {
            println("SKIPPED: this host runs $short at every indexed width, so no column mixture exists")
            return
        }
        val mixed = SparseMatrix.ofColumns(
            LONG_COLUMN,
            2,
            listOf(listOf(0 to 1.0), List(LONG_COLUMN) { row -> row to (row + 1.0) }),
        )

        val route = engine.routeOf(SparseMatrixOperation.Gemv, SparseCall(mixed))

        assertEquals(RouteKind.Composed, route.kind)
        assertEquals(listOf("$short/axpy", "$long/axpy"), route.components)
        assertTrue(!route.exactlyMeasurable, "a mixture is not an exact measurement of either component")
        assertContains(assertNotNull(route.reason), "straddle")
    }

    @Test
    fun `a transposed gemv names the ordered scalar dot its contract fixes`() {
        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar

        val route = engine.routeOf(
            SparseMatrixOperation.GemvTransposed,
            SparseCall(uniform(4, 4096)),
        )

        assertEquals(listOf("scalar/dotDense"), route.components, "a transposed CSC reduction is ordered")
        assertContains(assertNotNull(route.reason), "ordered scalar dot")
    }

    /**
     * A sparse operand on the right turns every update into a whole dense column, so a Level 1 kernel really
     * does run. It does not run for every stored entry, because the traversal skips the ones whose update is
     * a zero multiplier, and a route that claimed otherwise would be naming a kernel for work it never saw.
     */
    @Test
    fun `a sparse operand on the right is a composition of the dense leaf and the traversal`() {
        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar

        val route = engine.routeOf(
            SparseMatrixOperation.GemmDenseRight,
            SparseCall(uniform(2, 8), alpha = 1.0, beta = 1.0, updateRun = 4096),
        )

        val component = route.components.single()
        assertEquals(RouteKind.Composed, route.kind)
        assertTrue(component.endsWith("/axpy"), "right-side product reported $component")
        assertEquals(
            assertNotNull(engine.routeOf(DenseOperation.Axpy, 4096).implementation),
            component.substringBefore('/'),
            "the leaf must be the one a dense Level 1 axpy of that width reaches",
        )
    }

    @Test
    fun `a zero multiplier reports no work and only the destination scaling`() {
        val engine = BuiltinEngines.scalar
        val a = uniform(columns = 2, perColumn = 8)

        val scaled = engine.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(a, alpha = 0.0, beta = 0.5, destinationElements = 4096),
        )
        val filled = engine.routeOf(
            SparseMatrixOperation.TrsmLeft,
            SparseCall(a, alpha = 0.0, destinationElements = 4096),
        )

        assertEquals(RouteKind.NoWork, scaled.kind)
        assertEquals(listOf("scalar/scale"), scaled.components)
        assertEquals(RouteKind.NoWork, filled.kind)
        assertEquals(emptyList(), filled.components)
    }

    /**
     * A zero multiplier does not stop an operation whose result is a fresh structure. The contract says the
     * operand's values are not read, not that its positions are not found, so there is real work to report.
     */
    @Test
    fun `a zero multiplier still reports work for a structural result`() {
        val route = BuiltinEngines.scalar.routeOf(
            SparseMatrixOperation.AddScaled,
            SparseCall(uniform(2, 8), alpha = 0.0),
        )

        assertEquals(RouteKind.Direct, route.kind)
    }

    @Test
    fun `an empty operand reports no work`() {
        val empty = SparseMatrix.ofColumns(0, 0, emptyList())

        val route = BuiltinEngines.scalar.routeOf(SparseMatrixOperation.Symv, SparseCall(empty))

        assertEquals(RouteKind.NoWork, route.kind)
    }

    /**
     * An operand with plenty stored and a destination with no elements is still a call with nothing to do.
     * Only the extents can say so, which is why the descriptor carries them rather than the operand alone.
     */
    @Test
    fun `an empty destination or depth reports no work whatever the operand holds`() {
        val engine = BuiltinEngines.scalar
        val a = uniform(columns = 4, perColumn = 32)

        val emptyDestination = engine.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(a, alpha = 1.0, beta = 1.0, destinationElements = 0, depth = 32),
        )
        val emptyDepth = engine.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(a, alpha = 1.0, beta = 1.0, destinationElements = 4096, depth = 0),
        )
        val real = engine.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(
                a,
                alpha = 1.0,
                beta = 1.0,
                destinationElements = 4096,
                depth = 32,
                rightHandSides = 16,
            ),
        )

        assertEquals(RouteKind.NoWork, emptyDestination.kind, "a destination with no elements")
        assertEquals(RouteKind.NoWork, emptyDepth.kind, "a product over nothing")
        assertEquals(RouteKind.Direct, real.kind, "a product with work to do")
    }

    /**
     * A product whose facts leave out the right-hand side count is reported as undecided rather than as a
     * call with no panel.
     *
     * The count is what the grouping, the staging and the bodies all follow from, so a route derived without
     * it would be a confident answer about a call whose shape it does not know. Saying so is the difference
     * between a missing fact and a fact that says nothing runs.
     */
    @Test
    fun `a product with no right hand side count reports that its panels are underivable`() {
        val engine = BuiltinEngines.scalar
        val a = uniform(columns = 4, perColumn = 32)

        val route = engine.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(a, alpha = 1.0, beta = 1.0, destinationElements = 4096, depth = 32),
        )

        assertEquals(RouteKind.Composed, route.kind)
        assertTrue(route.components.none { it.endsWith("/sparse-rhs-scatter") }, route.toString())
        assertTrue(route.reason.orEmpty().contains("no right-hand side count"), route.toString())
    }

    /**
     * The same missing fact with no destination extent either, which is what a caller building the smallest
     * possible descriptor gives.
     *
     * A call with nothing to do has already returned by the time the panels are asked about, so what is
     * left is a call that will cut panels and a descriptor that does not say how many. Reporting a
     * confident portable answer there was the defect; the destination extent is not a stand-in for the
     * count, and a report reads the route's own flag rather than guessing from an empty component list.
     */
    @Test
    fun `a product with neither destination nor right hand side facts is reported unresolved`() {
        val engine = BuiltinEngines.scalar
        val a = uniform(columns = 4, perColumn = 32)

        val route = engine.routeOf(SparseMatrixOperation.GemmDense, SparseCall(a, alpha = 1.0))

        assertEquals(RouteKind.Composed, route.kind)
        assertEquals(false, route.resolved, route.toString())
        assertTrue(route.components.none { it.endsWith("/sparse-rhs-scatter") }, route.toString())
    }

    /** An unresolved route still names what the facts do settle, such as a destination scaling. */
    @Test
    fun `an unresolved route keeps the scaling its facts do settle`() {
        val engine = BuiltinEngines.scalar
        val a = uniform(columns = 4, perColumn = 32)

        val route = engine.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(a, alpha = 1.0, beta = -0.25, destinationElements = 4096, depth = 32),
        )

        assertEquals(false, route.resolved, route.toString())
        assertTrue(route.components.any { it.endsWith("/scale") }, route.toString())
    }

    /** The geometry a call resolved is in its reason, which is a scheduling choice and not a lane count. */
    @Test
    fun `a product names the grouping and the last group it cut`() {
        val engine = BuiltinEngines.scalar
        val a = uniform(columns = 4, perColumn = 32)

        val route = engine.routeOf(
            SparseMatrixOperation.GemmDense,
            SparseCall(
                a,
                alpha = 1.0,
                beta = 1.0,
                destinationElements = 4096,
                depth = 32,
                rightHandSides = 9,
            ),
        )

        val group = BuiltinEngines.scalar.panelKernels.executionGroup(PanelWork.SparseRightHandSides, 32, 9)
        assertEquals(group, route.executionGroup, "the route did not carry its grouping: $route")
        assertEquals(9 % group, route.executionTail, "the route did not carry its last group: $route")
        assertEquals(
            if (9 % group == 0) "@$group" else "@$group+${9 % group}",
            route.groupSuffix,
            "the published grouping",
        )
    }

    /**
     * An operation with no dense destination at all is not the same as one whose destination is empty, and a
     * descriptor that folded the two together would report real work as none.
     */
    @Test
    fun `an operation with no dense destination is not mistaken for an empty one`() {
        val route = BuiltinEngines.scalar.routeOf(
            SparseMatrixOperation.SyrkSparse,
            SparseCall(uniform(columns = 4, perColumn = 32), depth = 32),
        )

        assertEquals(RouteKind.Direct, route.kind)
    }

    /**
     * A triangular block with the triangle on the right names the dense kernel its column updates can reach.
     * It cannot promise the kernel ran: a stored zero coefficient skips its update, and so does a position
     * outside the selected triangle, so the route is composed and says which.
     */
    @Test
    fun `a triangle whose stored coefficients are all zero still names only a possible component`() {
        val engine = BuiltinEngines.scalar
        val zeros = SparseMatrix.ofColumns(3, 3, List(3) { j -> (j until 3).map { it to 0.0 } })

        val route = engine.routeOf(
            SparseMatrixOperation.TrsmRight,
            SparseCall(zeros, alpha = 1.0, destinationElements = 12, updateRun = 4),
        )

        assertEquals(RouteKind.Composed, route.kind)
        assertTrue(!route.exactlyMeasurable)
        assertContains(assertNotNull(route.reason), "stored zero coefficient")
    }

    @Test
    fun `preparing a snapshot reports the copy rather than an arithmetic kernel`() {
        val route = BuiltinEngines.scalar.routeOf(
            SparseMatrixOperation.Prepare,
            SparseCall(uniform(2, 8)),
        )

        assertEquals("portable-csc", route.implementation)
        assertEquals("spprepare", route.entryPoint)
        assertEquals(emptyList(), route.components)
        assertContains(assertNotNull(route.reason), "no arithmetic kernel runs")
    }
}
