package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseVector
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.koblas
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import com.eignex.koblas.vendor.RouteKind
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * A backend that records every window handed to it and answers with one of two bodies by length.
 *
 * Two bodies because one would make any claim true: a route naming the only implementation there is cannot
 * be wrong about which windows reached it. With a threshold in the middle of a triangular traversal's
 * lengths, a route that enumerated extents instead of windows names a body no window reached.
 */
private class RecordingPanels(private val group: Int, private val threshold: Int) : DensePanelKernels {
    val windows = ArrayList<Int>()

    override val name: String get() = "recording"

    override fun executionGroup(work: PanelWork, rows: Int, columns: Int): Int =
        if (columns <= 0) 1 else minOf(group, columns)

    override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String =
        if (rows >= threshold) "wide" else "narrow"

    override fun multiDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        windows.add(rows)
        PortablePanelKernels.multiDot(
            alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, beta, y, yOffset, yStride,
        )
    }

    override fun columnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        windows.add(rows)
        PortablePanelKernels.columnUpdate(
            alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset, yStride,
        )
    }

    override fun coupledUpdateDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        sums: DoubleArray,
        sumOffset: Int,
    ) {
        windows.add(rows)
        PortablePanelKernels.coupledUpdateDot(
            alpha, a, aOffset, lda, x, xOffset, rows, columns, y, yOffset,
            coefficients, coefficientOffset, sums, sumOffset,
        )
    }

    override fun rankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        coefficientStride: Int,
    ) {
        windows.add(rows)
        PortablePanelKernels.rankUpdate(
            alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns,
            coefficients, coefficientOffset, coefficientStride,
        )
    }
}

/**
 * What a dense matrix call says it executes, which has to follow from the call rather than from the engine.
 *
 * The question a route answers is not which backend was selected but which of its bodies this shape reaches,
 * and the two differ exactly where a window is too short for a vector or where a traversal's windows are not
 * all the same length.
 */
class DenseMatrixRouteTest {
    private val engines: List<KoblasEngine>
        get() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, BuiltinEngines.simd)

    @Test
    fun `a rectangular call names one panel and the grouping it will use`() {
        for (engine in engines) {
            val route = engine.denseRouteOf(DenseMatrixOperation.Gemv, DenseCall(64, 16, 1.0, 1.0))

            assertEquals(RouteKind.Direct, route.kind, engine.name)
            assertEquals("portable-dense", route.scheduling)
            assertEquals("gemv", route.entryPoint)
            assertEquals(listOf("${panelFor(engine, PanelWork.ColumnUpdate, 64)}/column-update"), route.components)
            assertEquals(engine.panelKernels.executionGroup(PanelWork.ColumnUpdate, 64, 16), route.executionGroup)
            assertTrue(route.exactlyMeasurable)
        }
    }

    @Test
    fun `a transposed matrix vector product is a different entry point from the untransposed one`() {
        val call = DenseCall(64, 16)
        val plain = koblas.denseRouteOf(DenseMatrixOperation.Gemv, call)
        val transposed = koblas.denseRouteOf(DenseMatrixOperation.GemvTransposed, call)

        assertEquals("gemv", plain.entryPoint)
        assertEquals("gemv-transposed", transposed.entryPoint)
        assertContains(plain.components.single(), "column-update")
        assertContains(transposed.components.single(), "multi-dot")
    }

    /**
     * A non-unit destination multiplier is a Level 1 `scale` and is named, where zero and one are neither a
     * kernel nor a read.
     */
    @Test
    fun `the destination scaling a call performs is a component like any other`() {
        val scaled = koblas.denseRouteOf(DenseMatrixOperation.Gemv, DenseCall(64, 16, 1.0, -0.25))
        val overwritten = koblas.denseRouteOf(DenseMatrixOperation.Gemv, DenseCall(64, 16, 1.0, 0.0))

        assertEquals(2, scaled.components.size, scaled.toString())
        assertContains(scaled.components.first(), "/scale")
        assertEquals(1, overwritten.components.size, overwritten.toString())
    }

    /**
     * A transposed product folds its destination multiplier into the panel that writes each output, so there
     * is no separate scaling kernel to name.
     */
    @Test
    fun `a transposed product applies beta inside its panel rather than through a kernel`() {
        val route = koblas.denseRouteOf(DenseMatrixOperation.GemvTransposed, DenseCall(64, 16, 1.0, -0.25))

        assertEquals(1, route.components.size, route.toString())
        assertContains(route.components.single(), "multi-dot")
    }

    /**
     * A triangular traversal's windows run from the full column down to nothing, so on a backend with a
     * vector body they do not all reach the same one and the route says composed rather than picking.
     */
    @Test
    fun `a traversal whose windows shrink is composed where its backend has more than one body`() {
        for (engine in engines) {
            val route = engine.denseRouteOf(DenseMatrixOperation.Symv, DenseCall(512, 512))
            val bodies = (1..512).map { engine.panelKernels.implementationFor(PanelWork.CoupledDotUpdate, it, 512) }

            if (bodies.distinct().size == 1) {
                assertEquals(RouteKind.Direct, route.kind, engine.name)
            } else {
                assertEquals(RouteKind.Composed, route.kind, engine.name)
                assertTrue(!route.exactlyMeasurable, engine.name)
                assertEquals(bodies.distinct().size, route.components.size, route.toString())
                assertContains(assertNotNull(route.reason), "shrink")
            }
        }
    }

    @Test
    fun `a call whose contract stops before the arithmetic reports no work`() {
        val noAlpha = koblas.denseRouteOf(DenseMatrixOperation.Gemv, DenseCall(64, 16, 0.0, -0.25))
        val empty = koblas.denseRouteOf(DenseMatrixOperation.Gemv, DenseCall(0, 16))
        val noDepth = koblas.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(64, 16, depth = 0))

        assertEquals(RouteKind.NoWork, noAlpha.kind)
        // The destination is still scaled, and that scaling is the one component such a call executes.
        assertContains(noAlpha.components.single(), "/scale")
        assertEquals(RouteKind.NoWork, empty.kind)
        assertEquals(emptyList(), empty.components)
        assertEquals(RouteKind.NoWork, noDepth.kind)
    }

    /** Level 3 is shared scalar traversal, and a row for it must not borrow a vector backend's name. */
    @Test
    fun `a level three call names no panel on any engine`() {
        for (engine in engines) {
            for (operation in listOf(
                DenseMatrixOperation.Gemm,
                DenseMatrixOperation.Gemmt,
                DenseMatrixOperation.Symm,
                DenseMatrixOperation.Syrk,
                DenseMatrixOperation.Syr2k,
                DenseMatrixOperation.Trsm,
                DenseMatrixOperation.Trmm,
            )) {
                val route = engine.denseRouteOf(operation, DenseCall(64, 16, 1.0, -0.25, depth = 32))

                assertEquals(emptyList(), route.components, "$operation on ${engine.name}")
                assertEquals(0, route.executionGroup, "$operation on ${engine.name}")
                assertEquals("portable-dense", route.implementation)
            }
        }
    }

    /** A strided shared vector is scalar work at any width, and the route is asked with that as a fact. */
    @Test
    fun `a strided operand reaches the portable body whatever the width`() {
        for (engine in engines) {
            val route = engine.denseRouteOf(
                DenseMatrixOperation.Syr,
                DenseCall(512, 512, contiguous = false),
            )

            assertTrue(
                route.components.all { it.startsWith(PortablePanelKernels.name) },
                "${engine.name} named ${route.components} for a strided operand",
            )
        }
    }

    /**
     * That the route's components are exactly the bodies the traversal's windows reach.
     *
     * Run against a backend that records what it is handed and answers by window length, so the claim is
     * checked against execution rather than restated. Every operation that schedules a panel is covered, on
     * both triangles and at groupings that leave an odd tail.
     */
    @Test
    fun `a route names the bodies the windows of its own traversal reach`() {
        val rng = Random(20260926)
        for (n in intArrayOf(1, 2, 4, 5, 8, 11)) {
            for (group in intArrayOf(1, 2, 3, 4)) {
                for (threshold in intArrayOf(1, 2, 4, 8)) {
                    for (lower in booleanArrayOf(true, false)) {
                        for (operation in PANEL_OPERATIONS) {
                            assertRouteMatchesExecution(operation, n, group, threshold, lower, rng)
                        }
                    }
                }
            }
        }
    }

    private fun assertRouteMatchesExecution(
        operation: DenseMatrixOperation,
        n: Int,
        group: Int,
        threshold: Int,
        lower: Boolean,
        rng: Random,
    ) {
        val panels = RecordingPanels(group, threshold)
        val blas = PortableDenseBlas(ScalarVectorKernels, panels)
        val a = randomMatrix(n, n, rng)
        for (i in 0 until n) a.values[i + i * n] = 2.0 + i % 3
        val x = randomVector(n, rng)
        val y = randomVector(n, rng)
        when (operation) {
            DenseMatrixOperation.Gemv -> blas.gemv(0.875, a, x, -0.25, y)
            DenseMatrixOperation.GemvTransposed -> blas.gemv(0.875, a, x, -0.25, y, transpose = true)
            DenseMatrixOperation.Symv -> blas.symv(0.875, a, x, -0.25, y, lower)
            DenseMatrixOperation.Ger -> blas.ger(0.875, x, y, a)
            DenseMatrixOperation.Syr -> blas.syr(0.875, DenseVector.wrap(x), a, lower)
            DenseMatrixOperation.Trmv -> blas.trmv(a, x, lower)
            DenseMatrixOperation.TrmvTransposed -> blas.trmv(a, x, lower, transpose = true)
            DenseMatrixOperation.Trsv -> blas.trsv(a, x, lower)
            DenseMatrixOperation.TrsvTransposed -> blas.trsv(a, x, lower, transpose = true)
            else -> error("$operation schedules no panel")
        }
        val executed = panels.windows.filter { it > 0 }
            .map { panels.implementationFor(PanelWork.MultiDot, it, 1) }
            .distinct()
        val route = blas.routeOf(operation, DenseCall(n, n, 0.875, -0.25, lower = lower))
        val named = route.components.filterNot { it.endsWith("/scale") }.map { it.substringBefore('/') }
        val context = "$operation n=$n group=$group threshold=$threshold lower=$lower"

        assertEquals(executed, named, "$context: windows ${panels.windows}")
        assertEquals(
            if (executed.size > 1) RouteKind.Composed else RouteKind.Direct,
            route.kind,
            context,
        )
    }

    /**
     * A triangle no wider than a lane block never reaches a vector body, because every window it cuts is
     * shorter than the one before it and the first is already short of the block.
     */
    @Test
    fun `a triangle narrower than the shortest vector window names only the portable body`() {
        val candidate = BuiltinEngines.simd ?: return println(
            "SKIPPED: no Vector API panel candidate on this host; the short-triangle route was not checked",
        )
        // Asked of the backend rather than derived from its grouping: a group of four columns and a lane
        // block of four rows are different numbers, and on a narrower species they do not agree at all.
        val shortest = shortestVectorWindow(candidate) ?: return println(
            "SKIPPED: this backend has no vector window; the short-triangle route was not checked",
        )
        val order = maxOf(2, shortest)

        for (operation in listOf(
            DenseMatrixOperation.Symv,
            DenseMatrixOperation.Trsv,
            DenseMatrixOperation.TrsvTransposed,
            DenseMatrixOperation.Trmv,
        )) {
            val route = candidate.denseRouteOf(operation, DenseCall(order, order))

            assertTrue(
                route.components.none { it.startsWith("simd") },
                "$operation over $order columns named ${route.components} where no window is that long",
            )
        }
    }

    /** An order of one leaves a traversal nothing but its corner, and the route may not name a panel. */
    @Test
    fun `a traversal with no window at all names no panel`() {
        for (engine in engines) {
            for (operation in listOf(
                DenseMatrixOperation.Symv,
                DenseMatrixOperation.Trsv,
                DenseMatrixOperation.Trmv,
                DenseMatrixOperation.TrsvTransposed,
            )) {
                val route = engine.denseRouteOf(operation, DenseCall(1, 1))

                assertEquals(emptyList(), route.components, "$operation of order one on ${engine.name}")
                assertEquals(RouteKind.Direct, route.kind)
                assertContains(assertNotNull(route.reason), "empty")
            }
        }
    }

    /**
     * A no-work transposed product still scales its destination through the kernel, because there is no panel
     * left to fold the multiplier into.
     */
    @Test
    fun `a transposed product with no work names the scaling it still performs`() {
        val route = koblas.denseRouteOf(DenseMatrixOperation.GemvTransposed, DenseCall(0, 64, 1.0, -0.25))

        assertEquals(RouteKind.NoWork, route.kind)
        assertContains(route.components.single(), "/scale")
    }

    private companion object {
        /** Longer than any window a backend distinguishes, so a search over it terminates. */
        const val LONG = 1024

        val PANEL_OPERATIONS = listOf(
            DenseMatrixOperation.Gemv,
            DenseMatrixOperation.GemvTransposed,
            DenseMatrixOperation.Symv,
            DenseMatrixOperation.Ger,
            DenseMatrixOperation.Syr,
            DenseMatrixOperation.Trmv,
            DenseMatrixOperation.TrmvTransposed,
            DenseMatrixOperation.Trsv,
            DenseMatrixOperation.TrsvTransposed,
        )
    }

    private fun panelFor(engine: KoblasEngine, work: PanelWork, rows: Int): String =
        engine.panelKernels.implementationFor(work, rows, 16)

    /**
     * The fewest rows at which this engine's backend answers with something other than the portable body,
     * or null where it never does.
     *
     * Asked of the backend, because the answer is a property of its arithmetic and not of its grouping. A
     * test that used the recommended group instead would be asserting the coincidence this seam exists to
     * keep apart.
     */
    private fun shortestVectorWindow(engine: KoblasEngine): Int? {
        val panels = engine.panelKernels
        val portable = panels.implementationFor(PanelWork.MultiDot, 1, 1)
        for (rows in 1..LONG) {
            if (panels.implementationFor(PanelWork.MultiDot, rows, 1) != portable) return rows
        }
        return null
    }
}
