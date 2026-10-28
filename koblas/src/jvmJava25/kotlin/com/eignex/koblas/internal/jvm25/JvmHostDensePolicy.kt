package com.eignex.koblas.internal.jvm25

import com.eignex.koblas.dense.DenseHostEligibility
import com.eignex.koblas.dense.DenseMatrixOperation

/** Whole Level 3 calls whose reuse can amortize the JVM binding's operand transfers. */
internal object JvmHostDensePolicy : DenseHostEligibility {
    /** A lower bound on every extent keeps thin products and few right-hand sides on JVM kernels. */
    private const val MINIMUM_EXTENT = 128

    override fun eligible(operation: DenseMatrixOperation, rows: Int, columns: Int, depth: Int): Boolean =
        rows >= MINIMUM_EXTENT && columns >= MINIMUM_EXTENT && depth >= MINIMUM_EXTENT && when (operation) {
            DenseMatrixOperation.Gemm, DenseMatrixOperation.Gemmt, DenseMatrixOperation.Symm,
            DenseMatrixOperation.Syrk, DenseMatrixOperation.Syr2k,
            DenseMatrixOperation.Trmm, DenseMatrixOperation.Trsm,
            -> true

            else -> false
        }
}
