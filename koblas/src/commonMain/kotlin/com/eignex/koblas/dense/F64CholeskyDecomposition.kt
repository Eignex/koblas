package com.eignex.koblas.dense

import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.requireShape

/**
 * A Cholesky factorization `A = L·Lᵀ`. Only the lower triangle of [l] is meaningful; the strict upper
 * triangle holds whatever the factorization left there.
 *
 * @property l the lower triangular factor `L`, square, with `A = L·Lᵀ`.
 */
public class F64CholeskyDecomposition @UnsafeKoblasApi constructor(
    @property:UnsafeKoblasApi public val l: F64DenseMatrix,
) {
    init {
        requireShape(l.rows == l.cols) { "cholesky factor must be square; got ${l.rows}x${l.cols}" }
    }

    /** The dimension of the factored matrix. */
    public val n: Int get() = l.rows

    /**
     * How many pivots [CholeskyPolicy.Regularize] replaced, or zero for a factorization of the input itself.
     *
     * A regularized factor is a genuine Cholesky of a nearby matrix rather than of the one handed in, and
     * without this there is no way to ask whether that happened: the policy raises nothing, and the factor it
     * returns looks like any other. A caller reading [logAbsDeterminant] off a factor with a nonzero count is
     * reading the perturbed matrix's determinant.
     */
    public var regularizations: Int = 0
        internal set
}
