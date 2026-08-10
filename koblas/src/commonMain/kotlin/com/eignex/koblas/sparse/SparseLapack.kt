package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SparseMatrix

/**
 * The sparse factorizations, the seam a host sparse solver plugs into.
 *
 * Named for symmetry with [com.eignex.koblas.dense.Lapack] rather than for accuracy — LAPACK has no
 * sparse half. The real world here is UMFPACK, KLU, SuperLU and CHOLMOD, and the symmetry is worth more
 * than the pedantry: a caller looking for the sparse counterpart of `factor` finds it where they expect.
 *
 * Returns [SparseFactorization] rather than a concrete [SparseLu], which is the design decision that makes
 * the seam usable at all — see that interface for why no sparse factor format is interchangeable.
 */
public interface SparseLapack : Backend {
    /**
     * Factorize the square [a] into something solvable, never null.
     *
     * A singular matrix comes back as a factorization reporting `singular`, matching the dense contract
     * where [com.eignex.koblas.dense.LuDecomposition] carries the flag. There is no nullable result to
     * unwrap and no exception on the ordinary path: a simplex whose basis went singular wants to inspect
     * and repair it, not catch something.
     *
     * With [equilibrate], rows are first scaled by a power of two so pivoting is better conditioned; the
     * scaling is undone transparently in the solves and the determinant.
     *
     * [dropTolerance] trades accuracy for sparsity: an entry the elimination produces whose magnitude is
     * below this fraction of the matrix's largest is discarded rather than stored, so the factors stay
     * sparser than the true `L·U`. The result is then an *incomplete* factorization — a solve against it is
     * an approximation, not a solve — which is why it defaults to zero and has to be asked for. It is worth
     * asking for when fill is the binding constraint and a residual around the tolerance is acceptable; on a
     * diagonally dominant matrix `1e-9` has been measured cutting fill threefold at a residual of `1e-11`
     * against `1e-15`. A backend that cannot honor it must fall back rather than ignore it.
     */
    public fun factor(
        a: SparseMatrix,
        equilibrate: Boolean = false,
        dropTolerance: Double = NO_DROP,
    ): SparseFactorization = SparseLu.factorCsc(a, equilibrate, dropTolerance)

    /**
     * Analyse a symmetric [a]'s pattern: the elimination tree and the nonzero pattern of `L`, reading no
     * values at all.
     *
     * The half of a symmetric factorization that depends only on the graph, separated because it is the
     * expensive half and it is reusable — every matrix with this pattern factors against one analysis. This is
     * the phase [factor] cannot have, since Markowitz pivoting picks its pivots from the values.
     *
     * A host backend with its own analysis (CHOLMOD's, say) overrides this and [ldl] together: an analysis is
     * only meaningful to the numeric phase that produced it.
     */
    public fun analyze(a: SparseMatrix, ordering: SparseOrdering = SparseOrdering.MinimumDegree): SparseSymbolic =
        SparseSymbolic.analyze(a, ordering)

    /**
     * `A = L·D·Lᵀ` for a symmetric [a], analysing the pattern first.
     *
     * Accepts an indefinite matrix by default, which is what separates this from [cholesky]: a KKT system has
     * negative pivots by construction and they are the answer rather than a failure. An exact zero pivot is
     * still singular, and comes back reporting the column it stopped at.
     *
     * To reuse the analysis across value updates, keep the returned factorization's `symbolic` and call
     * [SparseSymbolic.factorLdl] for the next set of values; this entry point is for the first factorization.
     */
    public fun ldl(
        a: SparseMatrix,
        policy: SparseLdlPolicy = SparseLdlPolicy.Indefinite,
        ordering: SparseOrdering = SparseOrdering.MinimumDegree,
    ): SparseFactorization = numericLdl(a, analyze(a, ordering), policy)

    /**
     * The Cholesky factorization of a symmetric positive-definite [a], as `L·D·Lᵀ` with every pivot positive.
     *
     * The same factorization [ldl] produces, held to a stricter contract: `L·√D` is the classical Cholesky
     * factor, available through `SparseLdl.choleskyFactor`. Keeping `D` rather than folding the square roots
     * into `L` is what lets one numeric kernel serve both spellings, at one divide per solve.
     *
     * A non-positive pivot throws by default, naming the column, as the dense `cholesky` does. Pass
     * [SparseLdlPolicy.Regularize] for the estimate-has-drifted case where a factorization that exists is
     * worth more than one that is exact.
     */
    public fun cholesky(
        a: SparseMatrix,
        policy: SparseLdlPolicy = SparseLdlPolicy.Strict,
        ordering: SparseOrdering = SparseOrdering.MinimumDegree,
    ): SparseFactorization = numericLdl(a, analyze(a, ordering), policy)
}
