package com.eignex.koblas.internal.jvm25.host

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.internal.host.DenseHostEligibility
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/** Whole Level 3 calls whose reuse can amortize the JVM binding's operand transfers. */
internal object JvmHostDensePolicy : DenseHostEligibility {
    /**
     * Extents that amortize copies, measured on Intel AVX-512 with transfers counted: from order 64 oneMKL
     * took about half the JVM time for the triangular routines and 0.6 to 0.85 of it for the products.
     */
    private const val MINIMUM_EXTENT = 64

    /** GEMMT still lost at order 64, and a borrowed view's extra packing was not part of the measurement. */
    private const val LARGER_MINIMUM_EXTENT = 128

    /** Deep products amortize transfers with fewer destination rows or columns. */
    private const val DEEP_PRODUCT_MINIMUM_SIDE = 16
    private const val DEEP_PRODUCT_MINIMUM_DEPTH = 1024

    // Very deep low-reuse inputs need their own transfer-cost evidence.
    private const val DEEP_PRODUCT_MAXIMUM_DEPTH = 4096

    override fun eligible(operation: DenseMatrixOperation, rows: Int, columns: Int, depth: Int): Boolean =
        largeCall(operation, rows, columns, depth, borrowed = false) ||
            (
                operation == DenseMatrixOperation.Gemm && rows >= DEEP_PRODUCT_MINIMUM_SIDE &&
                    columns >= DEEP_PRODUCT_MINIMUM_SIDE &&
                    depth in DEEP_PRODUCT_MINIMUM_DEPTH..DEEP_PRODUCT_MAXIMUM_DEPTH
                )

    // A view's extra packing was not part of the crossover measurements.
    override fun eligibleBorrowed(operation: DenseMatrixOperation, rows: Int, columns: Int, depth: Int): Boolean =
        largeCall(operation, rows, columns, depth, borrowed = true)

    private fun largeCall(
        operation: DenseMatrixOperation,
        rows: Int,
        columns: Int,
        depth: Int,
        borrowed: Boolean,
    ): Boolean {
        val minimum =
            if (borrowed || operation == DenseMatrixOperation.Gemmt) LARGER_MINIMUM_EXTENT else MINIMUM_EXTENT
        return rows >= minimum && columns >= minimum && depth >= minimum && when (operation) {
            DenseMatrixOperation.Gemm, DenseMatrixOperation.Gemmt, DenseMatrixOperation.Symm,
            DenseMatrixOperation.Syrk, DenseMatrixOperation.Syr2k,
            DenseMatrixOperation.Trmm, DenseMatrixOperation.Trsm,
            -> true

            else -> false
        }
    }
}
