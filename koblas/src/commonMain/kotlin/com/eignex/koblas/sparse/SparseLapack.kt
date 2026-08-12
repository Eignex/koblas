package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SparseMatrix

public interface SparseLapack : Backend {
    /**
     * Factorize the square [a] into something solvable. A singular matrix comes back as a factorization
     * reporting `singular` rather than as an exception.
     *
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
     * default; an exact zero pivot is still singular and reports the column it stopped at.
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
}
