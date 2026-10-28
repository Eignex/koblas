package com.eignex.koblas.internal.jvm25

import com.eignex.koblas.dense.DenseHostEligibility
import com.eignex.koblas.dense.DenseMatrixOperation

/** Whole Level 3 calls whose reuse can amortize the JVM binding's operand transfers. */
internal object JvmHostDensePolicy : DenseHostEligibility {
    /** Large extents amortize copies across every product family. */
    private const val MINIMUM_EXTENT = 128

    /** Deep products amortize transfers with fewer destination rows or columns. */
    private const val DEEP_PRODUCT_MINIMUM_SIDE = 16
    private const val DEEP_PRODUCT_MINIMUM_DEPTH = 1024

    // Very deep low-reuse inputs need their own transfer-cost evidence.
    private const val DEEP_PRODUCT_MAXIMUM_DEPTH = 16_384

    override fun eligible(operation: DenseMatrixOperation, rows: Int, columns: Int, depth: Int): Boolean =
        largeCall(operation, rows, columns, depth) ||
            (
                operation == DenseMatrixOperation.Gemm && rows >= DEEP_PRODUCT_MINIMUM_SIDE &&
                    columns >= DEEP_PRODUCT_MINIMUM_SIDE &&
                    depth in DEEP_PRODUCT_MINIMUM_DEPTH..DEEP_PRODUCT_MAXIMUM_DEPTH
                )

    // A view's extra packing was not part of the deep-product crossover measurements.
    override fun eligibleBorrowed(operation: DenseMatrixOperation, rows: Int, columns: Int, depth: Int): Boolean =
        largeCall(operation, rows, columns, depth)

    private fun largeCall(operation: DenseMatrixOperation, rows: Int, columns: Int, depth: Int): Boolean =
        rows >= MINIMUM_EXTENT && columns >= MINIMUM_EXTENT && depth >= MINIMUM_EXTENT && when (operation) {
            DenseMatrixOperation.Gemm, DenseMatrixOperation.Gemmt, DenseMatrixOperation.Symm,
            DenseMatrixOperation.Syrk, DenseMatrixOperation.Syr2k,
            DenseMatrixOperation.Trmm, DenseMatrixOperation.Trsm,
            -> true

            else -> false
        }
}
