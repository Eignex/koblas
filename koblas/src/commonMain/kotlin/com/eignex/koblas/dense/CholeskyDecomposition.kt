package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.requireShape

/**
 * A Cholesky factorization `A = L·Lᵀ` of a symmetric positive-definite matrix, holding the lower
 * triangular factor. Produced by [Lapack.cholesky]; consumed by [Lapack.solve] and [Lapack.invert].
 *
 * A type rather than the bare [DenseMatrix] this used to be, for the reason every other factorization
 * family already had one: `solveSpd(L, b)` accepted any matrix at all, so passing the unfactored `A` — or
 * the factor of a different matrix — compiled and returned silent nonsense. Every other factorization in
 * koblas is a distinct type its solve will not take anything else for, and this one no longer is the
 * exception.
 *
 * The factor stays reachable as [l] and is an ordinary [DenseMatrix], so nothing is hidden: a host backend
 * hands `l.data` across the FFI boundary exactly as before, and a caller who wants the entries reads them.
 * The wrapper costs one allocation per factorization and buys back the argument the type system could not
 * previously check.
 *
 * Only the lower triangle is meaningful. The strict upper triangle is left as whatever the factorization
 * put there, which is zero for koblas's own implementation and unspecified for a backend's.
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
