package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.requireSquare

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
     * A cheap pivot-quality estimate: `min(abs(U(k, k))) / max(abs(U(k, k)))`. A small value warns that
     * the factorization may be inaccurate; it is not a reciprocal condition-number estimate.
     */
    public val rcond: Double

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
}

/**
 * A sparse factorization of a simplex basis that can follow a replacement of one basis column.
 *
 * [replaceColumn] returns the factorization of the resulting basis, which supersedes this factorization. An
 * implementation may update its factors directly or rebuild them when the replacement would be too dense or
 * numerically unsafe to update.
 */
public interface F64BasisFactorization : F64SparseFactorization {
    /** The square basis matrix represented by this factorization. */
    public val basis: F64SparseMatrix

    /**
     * Replace [column] of [basis] with [entering] and return the factorization of the resulting basis, which
     * supersedes this factorization.
     *
     * [column] must name a column of [basis], and [entering] must have length [n].
     */
    public fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization
}

/** What [F64SparseDecompositions.factor] returns when no numerically acceptable pivot remains. */
public class F64SingularSparseFactorization(override val n: Int, override val failedAt: Int) : F64SparseFactorization {
    override val nnz: Int get() = 0

    override val rcond: Double get() = 0.0

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        throw singularFailure(failedAt, "solve")
}

/** The shapes [F64SparseFactorization.solveInto] requires of its right-hand side and its destination. */
public fun requireSolveShapes(n: Int, b: DoubleArray, out: DoubleArray) {
    requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
    requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
}

/**
 * A basis factorization over any [F64SparseDecompositions], for a backend that cannot update its own factors. A
 * replacement refactorizes the basis it produces, so the factors stay exact at the cost of a factorization
 * per replacement.
 *
 * Public because it is what a caller wanting a basis factorization from a backend that does not offer one
 * builds it from. BASICLU offers one; nothing else koblas binds does.
 */
public class F64RefactoringBasisFactorization(
    private val lu: F64SparseDecompositions,
    override val basis: F64SparseMatrix,
    private val factors: F64SparseFactorization,
) : F64BasisFactorization {
    init {
        requireSquare(basis, "factorBasis")
    }

    override val n: Int get() = factors.n

    override val failedAt: Int get() = factors.failedAt

    override val nnz: Int get() = factors.nnz

    override val rcond: Double get() = factors.rcond

    override fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization {
        val next = basis.withColumn(column, entering)
        return F64RefactoringBasisFactorization(lu, next, lu.factor(next))
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        factors.solveInto(b, out, transpose, workspace)
}
