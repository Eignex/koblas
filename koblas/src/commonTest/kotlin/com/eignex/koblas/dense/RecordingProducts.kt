@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.PortableEngine
import com.eignex.koblas.engineWithKernels
import com.eignex.koblas.internal.dense.OutputTriangle
import com.eignex.koblas.internal.dense.insideTriangle
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// Recording executed blocks lets checks follow the actual schedule while retaining its arithmetic.

/** One block a product scheduled, as the destination window it wrote and the scaling it carried. */
internal class ProductBlockCall(
    val rows: Int,
    val columns: Int,
    val depth: Int,
    val beta: Double,
    val destination: Int,
    /**
     * The array the block accumulated into, recorded because a block straddling the diagonal goes into a
     * tile of scratch and only its selected entries are merged out of it.
     */
    val into: DoubleArray,
)

/** A product backend that records every block it is handed and delegates the arithmetic. */
internal class RecordingProducts(private val delegate: KoblasEngine) : KoblasEngine() {
    override val implementations = setOf(EngineImplementation.Products)

    override fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> = delegate.implementationsFor(
        rows,
        columns,
        depth,
    )

    override fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean = delegate.packsProduct(
        rows,
        columns,
        depth,
    )

    override val tileRows get() = delegate.tileRows
    override val tileColumns get() = delegate.tileColumns

    /** Every block, in call order. */
    val blocks: MutableList<ProductBlockCall> = ArrayList()

    override val name: String get() = "recording(${delegate.name})"

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
 * That a triangle-selected product writes only its selected triangle, and that a block merged afterwards
 * accumulated into scratch without scaling it.
 *
 * The schedule cuts three kinds of block: one wholly in the selected triangle, which reaches the destination
 * directly; one straddling the diagonal, which accumulates into scratch so the merge can keep part of it;
 * and one wholly outside, which is never scheduled. Only running the product shows which each block was.
 */
internal fun assertTriangleProductStaysInTriangle(
    products: KoblasEngine,
    panels: KoblasEngine,
    n: Int,
    k: Int,
    lower: Boolean,
    transposeA: Boolean = false,
    transposeB: Boolean = false,
) {
    val rng = Random(20261101)
    val recorder = RecordingProducts(products)
    val blas = engineWithKernels(PortableEngine, panels, recorder)
    val a = randomMatrix(if (transposeA) k else n, if (transposeA) n else k, rng)
    val b = randomMatrix(if (transposeB) n else k, if (transposeB) k else n, rng)
    val c = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
    val context = "${n}x$k lower=$lower tA=$transposeA tB=$transposeB on ${products.name}"

    blas.gemmt(0.875, a, transposeA, b, transposeB, -0.25, c, lower)

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
