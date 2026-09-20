@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.vendor.RouteKind
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// A product route claims something about the blocks a call cuts, so checking it means recording those
// blocks. The recorder below delegates the arithmetic, so a call under it computes the same answer while
// saying what it was handed.

/** One block a product scheduled, as the destination window it wrote and the scaling it carried. */
internal class ProductBlockCall(
    val rows: Int,
    val columns: Int,
    val depth: Int,
    val beta: Double,
    val destination: Int,
    /**
     * The array the block accumulated into.
     *
     * Recorded because a selected-triangle schedule writes some blocks somewhere else: a block straddling
     * the diagonal goes into a tile of scratch and only its selected entries are merged out of it, and a
     * test that could not tell the two apart could not check that the merge is named.
     */
    val into: DoubleArray,
)

/** A product backend that records every block it is handed and delegates the arithmetic. */
internal class RecordingProducts(private val delegate: DenseProductKernels) : DenseProductKernels {
    /** Every block, in call order. */
    val blocks: MutableList<ProductBlockCall> = ArrayList()

    override val name: String get() = "recording(${delegate.name})"

    override val tileRows: Int get() = delegate.tileRows

    override val tileColumns: Int get() = delegate.tileColumns

    override fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> =
        delegate.implementationsFor(rows, columns, depth)

    override fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean =
        delegate.packsProduct(rows, columns, depth)

    override fun productBlock(
        alpha: Double,
        packedA: DoubleArray,
        aOffset: Int,
        aGroupStride: Int,
        packedB: DoubleArray,
        bOffset: Int,
        bGroupStride: Int,
        rows: Int,
        columns: Int,
        depth: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
    ) {
        blocks.add(ProductBlockCall(rows, columns, depth, beta, cOffset, c))
        delegate.productBlock(
            alpha, packedA, aOffset, aGroupStride, packedB, bOffset, bGroupStride,
            rows, columns, depth, beta, c, cOffset, ldc,
        )
    }
}

/**
 * The bodies one recorded block reaches, asked of the backend about the block it was really handed.
 *
 * Derived from the block the traversal produced rather than from the call's extents, which is the
 * distinction the whole recording exists for.
 */
private fun bodiesOf(products: DenseProductKernels, block: ProductBlockCall): List<String> =
    products.implementationsFor(block.rows, block.columns, block.depth)

/**
 * That a product's route names the tile its blocks reached, and that beta reached each of them once.
 *
 * Both halves come from running the product and watching it rather than from reasoning about the extents:
 * which window each block wrote, and which of the blocks writing a given window carried the caller's beta.
 * A depth cut into several blocks has to leave exactly one of them carrying it, and every other one
 * accumulating, which is the rule the whole of the blocking rests on.
 */
internal fun assertProductRouteNamesExecutedBlocks(
    products: DenseProductKernels,
    panels: DensePanelKernels,
    m: Int,
    n: Int,
    k: Int,
    transposeA: Boolean = false,
    transposeB: Boolean = false,
) {
    val rng = Random(20261020)
    val recorder = RecordingProducts(products)
    val blas = PortableDenseBlas(ScalarVectorKernels, panels, recorder)
    val a = randomMatrix(if (transposeA) k else m, if (transposeA) m else k, rng)
    val b = randomMatrix(if (transposeB) n else k, if (transposeB) k else n, rng)
    val c = DenseMatrix(m, n, DoubleArray(m * n) { rng.nextDouble(-1.0, 1.0) })
    val beta = -0.25
    val context = "${m}x${n}x$k tA=$transposeA tB=$transposeB on ${products.name}"

    blas.gemm(0.875, a, transposeA, b, transposeB, beta, c)

    val route = blas.routeOf(
        DenseMatrixOperation.Gemm,
        DenseCall(m, n, 0.875, beta, depth = k, transposeA = transposeA, transposeB = transposeB),
    )
    val tiles = route.components.filter { it.endsWith("/product-block") }.map { it.substringBefore('/') }
    if (recorder.blocks.isEmpty()) {
        assertTrue(tiles.isEmpty(), "$context: the route named $tiles and no block ran")
        return
    }
    val reached = recorder.blocks.flatMap { bodiesOf(products, it) }.distinct()
    assertEquals(reached, tiles, "$context: the route named $tiles")
    assertEquals(if (reached.size > 1) RouteKind.Composed else RouteKind.Direct, route.kind, context)

    val carried = recorder.blocks.groupBy { it.destination }
    for ((window, blocks) in carried) {
        val scaled = blocks.filter { it.beta != 1.0 }
        assertEquals(1, scaled.size, "$context: window $window took beta ${scaled.size} times")
        assertEquals(beta, scaled.single().beta, "$context: window $window took the wrong beta")
        assertEquals(blocks.first(), scaled.single(), "$context: beta did not go to the first depth block")
        assertEquals(k, blocks.sumOf { it.depth }, "$context: window $window did not cover the depth")
    }
}

/**
 * That a triangle-selected product's route names the bodies its blocks reached and the merge they needed.
 *
 * The schedule cuts three kinds of block and the route has to distinguish them: one wholly in the selected
 * triangle, which reaches the destination directly; one straddling the diagonal, which accumulates into a
 * tile of scratch so that the merge can keep part of it; and one wholly outside, which is never scheduled.
 * Only running the product shows which of the three each block was, which is why the destination array is
 * recorded and compared with the call's own.
 */
internal fun assertTriangleProductRouteNamesExecutedBlocks(
    products: DenseProductKernels,
    panels: DensePanelKernels,
    n: Int,
    k: Int,
    lower: Boolean,
    transposeA: Boolean = false,
    transposeB: Boolean = false,
) {
    val rng = Random(20261101)
    val recorder = RecordingProducts(products)
    val blas = PortableDenseBlas(ScalarVectorKernels, panels, recorder)
    val a = randomMatrix(if (transposeA) k else n, if (transposeA) n else k, rng)
    val b = randomMatrix(if (transposeB) n else k, if (transposeB) k else n, rng)
    val c = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
    val context = "${n}x$k lower=$lower tA=$transposeA tB=$transposeB on ${products.name}"

    blas.gemmt(0.875, a, transposeA, b, transposeB, -0.25, c, lower)

    val route = blas.routeOf(
        DenseMatrixOperation.Gemmt,
        DenseCall(n, n, 0.875, -0.25, depth = k, lower = lower, transposeA = transposeA, transposeB = transposeB),
    )
    val tiles = route.components.filter { it.endsWith("/product-block") }.map { it.substringBefore('/') }
    val reached = recorder.blocks.flatMap { bodiesOf(products, it) }.distinct()
    assertEquals(reached, tiles, "$context: the route named $tiles")

    val merged = recorder.blocks.any { it.into !== c.values }
    assertEquals(
        merged,
        route.components.any { it == "$TRIANGLE_SELECTION/triangle-tile" },
        "$context: the merge ran=$merged and the route said otherwise in ${route.components}",
    )
    for (block in recorder.blocks) {
        if (block.into !== c.values) {
            assertEquals(0.0, block.beta, "$context: a tile merged afterwards scaled the scratch it wrote")
        }
    }
    val outside = recorder.blocks.none { it.into === c.values && unselectedWindow(it, n, lower) }
    assertTrue(outside, "$context: a block wrote into the triangle the call does not select")
}

/** Whether a block written straight into the destination lies anywhere outside the selected triangle. */
private fun unselectedWindow(block: ProductBlockCall, n: Int, lower: Boolean): Boolean {
    val row = block.destination % n
    val column = block.destination / n
    return !insideTriangle(
        row,
        block.rows,
        column,
        block.columns,
        if (lower) OutputTriangle.Lower else OutputTriangle.Upper,
    )
}
