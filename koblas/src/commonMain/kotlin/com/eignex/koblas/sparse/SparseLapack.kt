package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace

/** Sparse factorizations as a backend half. */
public interface SparseLapack : Backend {
    /**
     * Factorize the square [a] into something solvable. A singular matrix comes back as a factorization
     * reporting `singular` rather than as an exception.
     *
     * @param a the square matrix to factorize.
     * @param equilibrate scale rows by a power of two first; the solves and the determinant undo it.
     * @param dropTolerance discard produced entries this far below the largest magnitude, giving an
     *   incomplete factorization.
     */
    public fun factor(
        a: SparseMatrix,
        equilibrate: Boolean = false,
        dropTolerance: Double = NO_DROP,
    ): SparseFactorization = SparseLu.factorCsc(a, equilibrate, dropTolerance)

    /**
     * Analyse a symmetric [a]'s pattern, the elimination tree and the nonzero pattern of L, reading no
     * values. Reusable, so every matrix with this pattern factors against one analysis.
     */
    public fun analyze(a: SparseMatrix, ordering: SparseOrdering = SparseOrdering.MinimumDegree): SparseSymbolic =
        SparseSymbolic.analyze(a, ordering)

    /**
     * `A = L·D·Lᵀ` for a symmetric [a], analysing the pattern first. Accepts an indefinite matrix by
     * default, and under that default an exact zero pivot is singular and reports the column it stopped at.
     */
    public fun ldl(
        a: SparseMatrix,
        policy: SparseLdlPolicy = SparseLdlPolicy.Indefinite,
        ordering: SparseOrdering = SparseOrdering.MinimumDegree,
    ): SparseFactorization = numericLdl(a, analyze(a, ordering), policy)

    /**
     * The Cholesky factorization of a symmetric positive-definite [a], as `L·D·Lᵀ` with every pivot
     * positive. The classical factor `L·√D` is available through `SparseLdl.choleskyFactor`.
     *
     * @throws com.eignex.koblas.NotPositiveDefinite on a non-positive pivot, naming the column.
     */
    public fun cholesky(
        a: SparseMatrix,
        policy: SparseLdlPolicy = SparseLdlPolicy.Strict,
        ordering: SparseOrdering = SparseOrdering.MinimumDegree,
    ): SparseFactorization = numericLdl(a, analyze(a, ordering), policy)

    /**
     * Solve `A·x = b` from [f] into [out], `Aᵀ·x = b` when [transpose]. The work belongs to the
     * factorization; this is here so the seam reads the same from the sparse side as [com.eignex.koblas
     * .dense.Lapack.solveInto] does from the dense one.
     */
    public fun solveInto(
        f: SparseFactorization,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray = f.solveInto(b, out, transpose, workspace)

    /** [solveInto] into a fresh vector. */
    public fun solve(f: SparseFactorization, b: DoubleArray, transpose: Boolean = false): DoubleArray =
        f.solve(b, transpose)
}
