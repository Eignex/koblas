package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.koblas
import com.eignex.koblas.vendor.RouteKind
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * What a dense matrix call says it executes, which has to follow from the call rather than from the engine.
 *
 * The question a route answers is not which backend was selected but which of its bodies this shape reaches,
 * and the two differ exactly where a window is too short for a vector or where a traversal's windows are not
 * all the same length.
 */
class DenseMatrixRouteTest {
    private val engines: List<KoblasEngine>
        get() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)

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
     * A triangular traversal's windows run from the full column down to nothing, so on a backend with more
     * than one body they need not all reach the same one, and the route says which of the two it is.
     *
     * Which it is depends on the schedule and not on the order. A symmetric traversal grouped by two cuts
     * only even windows at an even order, so a backend whose shortest vector window is two serves every one
     * of them and the call really is direct; make the order odd and the last window is one long and the same
     * call is a composition. Both are checked against what the traversal did rather than against a length
     * the schedule never produces.
     */
    @Test
    fun `a symmetric traversal is composed exactly when its own windows reach more than one body`() {
        for (engine in engines) {
            assertRouteNamesExecutedBodies(engine.panelKernels)
        }
    }

    /** A composition says so in its reason and refuses to be read as an exact measurement. */
    @Test
    fun `a composed route declines to stand for either of the bodies it names`() {
        val composed = engines.asSequence()
            .map { it.denseRouteOf(DenseMatrixOperation.Symv, DenseCall(513, 513)) }
            .firstOrNull { it.kind == RouteKind.Composed }
            ?: return println("SKIPPED: every backend here has one body, so no call is a composition")

        assertTrue(!composed.exactlyMeasurable, composed.toString())
        assertTrue(composed.components.size > 1, composed.toString())
        assertContains(assertNotNull(composed.reason), "shrink")
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
     * The same check at groupings and body thresholds no real backend here produces.
     *
     * A backend with one body cannot make the claim false, and a real one's threshold sits where its lane
     * block is. Overriding both reaches the shapes in between, where a body changes part way through a
     * schedule, which is the case a route derived from extents rather than windows gets wrong.
     */
    @Test
    fun `a route follows the schedule at groupings and thresholds no backend here has`() {
        for (group in intArrayOf(1, 2, 3, 4)) {
            for (threshold in intArrayOf(1, 2, 3, 4, 8)) {
                assertRouteNamesExecutedBodies(PortablePanelKernels, group, threshold, SHORT_ROUTE_ORDERS)
            }
        }
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
