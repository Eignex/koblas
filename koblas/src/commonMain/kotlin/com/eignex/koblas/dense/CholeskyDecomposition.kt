package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.requireShape

/**
 * A Cholesky factorization `A = L·Lᵀ`. Only the lower triangle of [l] is meaningful; the strict upper
 * triangle holds whatever the factorization left there.
 *
 * @property l the lower triangular factor `L`, square, with `A = L·Lᵀ`.
 */
public class CholeskyDecomposition(public val l: DenseMatrix) {
    init {
        requireShape(l.rows == l.cols) { "cholesky factor must be square; got ${l.rows}x${l.cols}" }
    }

    /** The dimension of the factored matrix. */
    public val n: Int get() = l.rows
}
