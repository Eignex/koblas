package com.eignex.koblas.sparse

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.Workspace
import com.eignex.koblas.singularFailure

/** A factorization held for reuse against further right-hand sides. */
public interface F64SparseFactorization {
    /** The dimension of the factored matrix. */
    public val n: Int

    /**
     * The pivot position that had no numerically acceptable candidate, or [NOT_SINGULAR] when the
     * factorization succeeded.
     */
    public val failedAt: Int

    /**
     * Whether the factorization failed for want of a numerically acceptable pivot. Solving against one
     * throws [com.eignex.koblas.SingularMatrix] rather than answering with infinities.
     */
    public val singular: Boolean get() = failedAt != NOT_SINGULAR

    /** Nonzeros in the factors, the fill. Zero for a singular factorization, which has none. */
    public val nnz: Int

    /**
     * Solve `B x = b`, or `Bᵀ x = b` when [transpose], into [out], which is returned. Allocates nothing
     * when given a [workspace]. [out] may be [b].
     */
    public fun solveInto(
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray

    /** Solve `B x = b`, or `Bᵀ x = b` when [transpose], into a fresh result. */
    public fun solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = solveInto(b, DoubleArray(n), transpose)

    /** `det(B)`, or exactly `0.0` when [singular]. */
    public fun determinant(): Double
}

/** What [F64SparseLapack.factor] returns when no numerically acceptable pivot remains. */
public class F64SingularSparseFactorization(override val n: Int, override val failedAt: Int) : F64SparseFactorization {
    override val nnz: Int get() = 0

    override fun determinant(): Double = 0.0

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        throw singularFailure(failedAt, "solve")
}
