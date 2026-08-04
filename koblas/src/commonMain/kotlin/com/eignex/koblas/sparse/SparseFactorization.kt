package com.eignex.koblas.sparse

import com.eignex.koblas.Workspace

/**
 * A factored sparse matrix you can solve against, whoever produced it.
 *
 * An interface rather than a class, and that is the whole point. The dense side passes a
 * [com.eignex.koblas.dense.LuDecomposition] between backends because LAPACK's packed formats *are* a
 * standard: a factorization from OpenBLAS solves on the reference and back again.
 *
 * Sparse has no such format. UMFPACK hands back a `void **Numeric`, KLU a `klu_numeric *`, CHOLMOD a
 * `cholmod_factor *` — every host solver keeps its factors in a private structure it will not describe,
 * each bound to the library that made it. A seam demanding a concrete [SparseLu] could therefore never
 * admit one.
 *
 * So [SparseLapack.factor] returns this, [SparseLu] is the portable implementation, and a host backend
 * would add another holding its own handle. It also keeps `SparseLu`'s internal arrays out of the public
 * seam, which they had no business being part of.
 *
 * @see com.eignex.koblas.dense.LuDecomposition the dense counterpart, a class precisely because its
 *   format is shared.
 */
interface SparseFactorization {
    /** The dimension of the factored matrix. */
    val n: Int

    /**
     * Whether the factorization failed for want of a numerically acceptable pivot.
     *
     * Solving against a singular factorization is not meaningful, matching the dense contract. Unlike the
     * dense case it *throws* rather than returning nonsense: a dense `getrf` completes and leaves partial
     * factors behind, so garbage is what LAPACK itself would produce, whereas Markowitz pivoting aborts
     * with nothing to solve with. Returning silence would be a lie about having factors.
     */
    val singular: Boolean

    /** Nonzeros in the factors — the fill. Zero for a singular factorization, which has none. */
    val nnz: Int

    /**
     * Solve `B x = b`, or `Bᵀ x = b` when [transpose], into [out], which is returned.
     *
     * The two directions are a simplex's FTRAN and BTRAN. Allocates nothing when given a [workspace], so
     * an iteration that owns its destination runs without touching the collector. [out] may be [b].
     */
    fun solveInto(
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray

    /** Solve `B x = b`, or `Bᵀ x = b` when [transpose], into a fresh result. */
    fun solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = solveInto(b, DoubleArray(n), transpose)

    /** `det(B)`, or exactly `0.0` when [singular]. The counterpart of `LuDecomposition.determinant`. */
    fun determinant(): Double
}

/**
 * What [SparseLapack.factor] returns when no numerically acceptable pivot remains.
 *
 * Markowitz pivoting stops at the failing column with a partially eliminated matrix and no usable
 * factors, so there is nothing to hand back — and fabricating a [SparseLu] whose buffers were never
 * filled would produce an object that answers `solve` with silent nonsense. This reports what is true
 * instead: singular, no fill, zero determinant, and an exception if solved against. [failedAt] is the
 * pivot position that had no candidate, which is more actionable than a null ever was — a simplex can use
 * it to pick which column to replace.
 */
class SingularSparseFactorization(override val n: Int, val failedAt: Int) : SparseFactorization {
    override val singular: Boolean get() = true

    override val nnz: Int get() = 0

    override fun determinant(): Double = 0.0

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        error(
            "cannot solve against a singular factorization: no acceptable pivot at position $failedAt of $n. " +
                "Check `singular` before solving; repair the matrix and factor again.",
        )
}
