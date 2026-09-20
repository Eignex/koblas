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
 *
 * These ask the exact built-in engines rather than the platform default. What they are about is this
 * library's own dense schedule and the bodies it reaches, and a default is a policy: Kotlin/Native's hands a
 * call of some of these shapes to an installed library instead, which is a different question and is asked
 * where that policy lives. Asking the exact engines also makes the answers the same on every host, rather
 * than depending on what happens to be installed on one.
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
        for (engine in engines) {
            val plain = engine.denseRouteOf(DenseMatrixOperation.Gemv, call)
            val transposed = engine.denseRouteOf(DenseMatrixOperation.GemvTransposed, call)

            assertEquals("gemv", plain.entryPoint, engine.name)
            assertEquals("gemv-transposed", transposed.entryPoint, engine.name)
            assertContains(plain.components.single(), "column-update")
            assertContains(transposed.components.single(), "multi-dot")
        }
    }

    /**
     * A non-unit destination multiplier is a Level 1 `scale` and is named, where zero and one are neither a
     * kernel nor a read.
     */
    @Test
    fun `the destination scaling a call performs is a component like any other`() {
        for (engine in engines) {
            val scaled = engine.denseRouteOf(DenseMatrixOperation.Gemv, DenseCall(64, 16, 1.0, -0.25))
            val overwritten = engine.denseRouteOf(DenseMatrixOperation.Gemv, DenseCall(64, 16, 1.0, 0.0))

            assertEquals(2, scaled.components.size, scaled.toString())
            assertContains(scaled.components.first(), "/scale")
            assertEquals(1, overwritten.components.size, overwritten.toString())
        }
    }

    /**
     * A transposed product folds its destination multiplier into the panel that writes each output, so there
     * is no separate scaling kernel to name.
     */
    @Test
    fun `a transposed product applies beta inside its panel rather than through a kernel`() {
        for (engine in engines) {
            val route = engine.denseRouteOf(DenseMatrixOperation.GemvTransposed, DenseCall(64, 16, 1.0, -0.25))

            assertEquals(1, route.components.size, route.toString())
            assertContains(route.components.single(), "multi-dot")
        }
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

    /**
     * Every structured Level 3 routine names the bodies it reaches, and none of them is an engine's name.
     *
     * The scheduling is this library's own on every engine, and what runs inside its windows is the
     * selected backends' and nothing else. A component naming the engine would be the claim these routes
     * exist to prevent: that a Vector API Level 1 selection makes a matrix routine vectorised.
     */
    @Test
    fun `a structured level three call names the bodies it reaches and never the engine`() {
        for (engine in engines) {
            val bodies = engine.productKernels.implementationsFor(64, 16, 32) +
                engine.triangularKernels.implementationsFor(16, 16, true) +
                engine.triangularKernels.implementationsFor(16, 16, false)
            for (operation in listOf(
                DenseMatrixOperation.Gemmt,
                DenseMatrixOperation.Symm,
                DenseMatrixOperation.Syrk,
                DenseMatrixOperation.Syr2k,
                DenseMatrixOperation.Trsm,
                DenseMatrixOperation.Trmm,
            )) {
                val route = engine.denseRouteOf(
                    operation,
                    DenseCall(64, 64, 1.0, -0.25, depth = 64),
                )

                assertEquals("portable-dense", route.scheduling, "$operation on ${engine.name}")
                assertTrue(route.components.isNotEmpty(), "$operation on ${engine.name} named nothing")
                assertTrue(
                    route.components.none { it.substringBefore('/') == engine.name },
                    "$operation on ${engine.name} named the engine: ${route.components}",
                )
                val named = route.components.map { it.substringBefore('/') }
                assertTrue(
                    named.any { it in bodies || it.startsWith("portable-") || it.endsWith("-panel") },
                    "$operation on ${engine.name} named ${route.components}",
                )
            }
        }
    }

    /**
     * A triangular matrix call whose order fits one diagonal block schedules no product between blocks, and
     * one that does not schedules both.
     *
     * Two calls of the same operation on the same engine, differing only in an extent, reaching different
     * components. That is the property the reason strings claim and the one a route built from the operation
     * name alone would get wrong.
     */
    @Test
    fun `a triangular solve names a product only where its order needs more than one block`() {
        for (engine in engines) {
            val single = engine.denseRouteOf(DenseMatrixOperation.Trsm, DenseCall(8, 4, depth = 8))
            val several = engine.denseRouteOf(
                DenseMatrixOperation.Trsm,
                DenseCall(4 * TRIANGULAR_DIAGONAL_BLOCK, 64, depth = 4 * TRIANGULAR_DIAGONAL_BLOCK),
            )

            assertTrue(
                single.components.none { it.endsWith("/product-block") || it.endsWith("/column-update") },
                "${engine.name} named a product in ${single.components} for an order inside one block",
            )
            assertTrue(
                single.components.any { it.endsWith("/diagonal-solve") },
                "${engine.name} named ${single.components} and no substitution",
            )
            assertContains(assertNotNull(single.reason), "no product runs between them")
            assertTrue(
                several.components.any { it.endsWith("/product-block") || it.endsWith("/column-update") },
                "${engine.name} named ${several.components} for an order of several blocks",
            )
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

    /** A product large enough to be packed names its tile and the two panels it packs into. */
    @Test
    fun `a packed product names the packing and the tile`() {
        for (engine in engines) {
            val route = engine.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(64, 64, 0.875, -0.25, depth = 64))

            assertEquals(RouteKind.Direct, route.kind, engine.name)
            assertEquals("gemm", route.entryPoint)
            assertEquals(
                listOf("portable-pack/right-panel", "portable-pack/left-panel") + wholeTileBody(engine),
                route.components,
                engine.name,
            )
            // A grouping is a panel's number, and a blocked product schedules no panel.
            assertEquals(0, route.executionGroup, engine.name)
        }
    }

    /**
     * A product too small to pay for a copy names the panel it runs on, which depends on the left transpose.
     *
     * The two are different arithmetic over the same numbers: transposed, a destination entry is a reduction
     * down a stored column, and untransposed, a destination column is those columns accumulated into it.
     */
    @Test
    fun `a small product names the panel its transpose flags reach`() {
        for (engine in engines) {
            val plain = engine.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(5, 4, depth = 6))
            val transposed = engine.denseRouteOf(
                DenseMatrixOperation.Gemm,
                DenseCall(5, 4, depth = 6, transposeA = true),
            )

            assertTrue(plain.components.any { it.endsWith("/column-update") }, "${engine.name} $plain")
            assertTrue(transposed.components.any { it.endsWith("/multi-dot") }, "${engine.name} $transposed")
            assertContains(assertNotNull(plain.reason), "packing")
        }
    }

    /**
     * A transposed right operand leaves the vector a reduction shares across its columns strided, and what
     * the route does about that is the question here.
     *
     * Two destination columns, which is fewer than any tile is wide, so this product is never packed however
     * long its other extents are; what varies below is only what the reduction's shared vector looks like.
     */
    @Test
    fun `a small transposed product names the body its coefficient stride reaches`() {
        for (engine in engines) {
            val portable = PortablePanelKernels.name
            val vectorBody = engine.panelKernels.implementationFor(PanelWork.MultiDot, 64, 4)
            val gathered = engine.denseRouteOf(
                DenseMatrixOperation.Gemm,
                DenseCall(64, 2, depth = 64, transposeA = true, transposeB = true),
            )
            val tooFewRows = engine.denseRouteOf(
                DenseMatrixOperation.Gemm,
                DenseCall(2, 2, depth = 64, transposeA = true, transposeB = true),
            )
            val oneColumn = engine.denseRouteOf(
                DenseMatrixOperation.Gemm,
                DenseCall(64, 1, depth = 64, transposeA = true, transposeB = true),
            )
            val adjacent = engine.denseRouteOf(
                DenseMatrixOperation.Gemm,
                DenseCall(64, 2, depth = 64, transposeA = true),
            )

            // A strided column with rows enough to read it back is gathered once, on a backend where that
            // changes which body the reduction reaches; a backend with one body gains nothing and copies
            // nothing.
            assertEquals(
                vectorBody != portable,
                gathered.components.any { it == "portable-pack/right-column" },
                "${engine.name} $gathered",
            )
            assertEquals(listOf("$vectorBody/multi-dot"), gathered.components.takeLast(1), "$gathered")
            // Too few destination rows to read a copied column back, so the strided body is what runs.
            assertTrue(
                tooFewRows.components.none { it == "portable-pack/right-column" },
                "${engine.name} gathered a column two destination rows would read: $tooFewRows",
            )
            assertEquals(listOf("$portable/multi-dot"), tooFewRows.components, "$tooFewRows")
            // One destination column leaves a transposed right operand adjacent already, so neither a copy
            // nor the portable body is the right answer for it.
            assertTrue(
                oneColumn.components.none { it == "portable-pack/right-column" },
                "${engine.name} gathered a column that was already adjacent: $oneColumn",
            )
            assertEquals(listOf("$vectorBody/multi-dot"), oneColumn.components, "$oneColumn")
            assertEquals(listOf("$vectorBody/multi-dot"), adjacent.components, "$adjacent")
        }
    }

    /**
     * A column update shares its destination strip, which this route hands over adjacent whatever the right
     * operand's transpose is, so a transposed right operand does not make it scalar work.
     */
    @Test
    fun `an untransposed small product reaches the same body whichever way its right operand is stored`() {
        for (engine in engines) {
            val plain = engine.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(64, 2, depth = 64))
            val transposed = engine.denseRouteOf(
                DenseMatrixOperation.Gemm,
                DenseCall(64, 2, depth = 64, transposeB = true),
            )

            assertEquals(
                listOf("${engine.panelKernels.implementationFor(PanelWork.ColumnUpdate, 64, 4)}/column-update"),
                plain.components,
                engine.name,
            )
            assertEquals(plain.components, transposed.components, engine.name)
        }
    }

    /** A retained panel is not packed again, and the route names only the packing a call still performs. */
    @Test
    fun `a product over retained panels names only the packing it still does`() {
        val call = DenseCall(64, 64, 0.875, -0.25, depth = 64)
        for (engine in engines) {
            val tile = wholeTileBody(engine)
            val both = engine.denseRouteOf(DenseMatrixOperation.GemmPacked, call)
            val left = engine.denseRouteOf(DenseMatrixOperation.GemmPackedLeft, call)
            val right = engine.denseRouteOf(DenseMatrixOperation.GemmPackedRight, call)

            assertEquals(tile, both.components, engine.name)
            assertEquals(listOf("portable-pack/right-panel") + tile, left.components, engine.name)
            assertEquals(listOf("portable-pack/left-panel") + tile, right.components, engine.name)
            assertEquals("gemm-packed", both.entryPoint)
            assertEquals("gemm-packed-left", left.entryPoint)
            assertEquals("gemm-packed-right", right.entryPoint)
        }
    }

    /**
     * The product route against the blocks the product really cut, on every built-in backend.
     *
     * The shapes are derived from the tile geometry this machine resolved, because what a route has to get
     * right moves with it: a destination of two whole tiles reaches one body, one of a tile and a single row
     * reaches whatever the remainder goes to as well, and a product long enough to be cut on each axis in
     * turn reaches its bodies across several blocks. One shape is too small to be packed at all, so the
     * assertion also covers a route that names no block because none ran.
     */
    @Test
    fun `a product route names the tiles its blocks reached and carries beta once`() {
        for (engine in engines) {
            val tile = engine.productKernels
            for ((m, n, k) in listOf(
                Triple(2 * tile.tileRows, 4 * tile.tileColumns, 512),
                Triple(tile.tileRows + 1, 4 * tile.tileColumns, 512),
                Triple(tile.tileRows, 4 * tile.tileColumns, PRODUCT_BLOCK_DEPTH * 2 + 17),
                Triple(PRODUCT_BLOCK_ROWS + tile.tileRows, 4 * tile.tileColumns, 64),
                Triple(2 * tile.tileRows, PRODUCT_BLOCK_COLUMNS + tile.tileColumns, 64),
                Triple(37, 29, 41),
                Triple(5, 4, 6),
            )) {
                for (transposeA in booleanArrayOf(false, true)) {
                    assertProductRouteNamesExecutedBlocks(
                        engine.productKernels,
                        engine.panelKernels,
                        m,
                        n,
                        k,
                        transposeA,
                    )
                }
            }
        }
    }

    /**
     * A destination of whole tiles names one body, and one with a row left over names the remainder's too.
     *
     * The difference is a property of the backend rather than of the route, so what is asserted is that the
     * route agrees with what the backend says about a block of that shape, on whichever backend this is.
     */
    @Test
    fun `a block whose rows fill its tiles is direct and one with a remainder is composed`() {
        for (engine in engines) {
            val tile = engine.productKernels
            val n = 4 * tile.tileColumns
            val whole = engine.denseRouteOf(
                DenseMatrixOperation.Gemm,
                DenseCall(2 * tile.tileRows, n, depth = 512),
            )
            val remainder = engine.denseRouteOf(
                DenseMatrixOperation.Gemm,
                DenseCall(tile.tileRows + 1, n, depth = 512),
            )
            val wholeBodies = tile.implementationsFor(2 * tile.tileRows, n, 512)
            val remainderBodies = tile.implementationsFor(tile.tileRows + 1, n, 512)

            assertEquals(
                wholeBodies.map { "$it/product-block" },
                whole.components.filter { it.endsWith("/product-block") },
                "${engine.name} $whole",
            )
            assertEquals(
                remainderBodies.map { "$it/product-block" },
                remainder.components.filter { it.endsWith("/product-block") },
                "${engine.name} $remainder",
            )
            assertEquals(
                if (wholeBodies.size > 1) RouteKind.Composed else RouteKind.Direct,
                whole.kind,
                "${engine.name} $whole",
            )
            assertEquals(
                if (remainderBodies.size > 1) RouteKind.Composed else RouteKind.Direct,
                remainder.kind,
                "${engine.name} $remainder",
            )
        }
    }

    private companion object {
        /** Longer than any window a backend distinguishes, so a search over it terminates. */
        const val LONG = 1024
    }

    /**
     * The route components a block whose rows fill whole tiles produces on this engine.
     *
     * Asked of the backend about a block of exactly that shape, so a backend that serves whole tiles with
     * two bodies would be described as it is rather than as this test assumed.
     */
    private fun wholeTileBody(engine: KoblasEngine): List<String> {
        val products = engine.productKernels
        return products.implementationsFor(products.tileRows, products.tileColumns, 64)
            .map { "$it/product-block" }
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
