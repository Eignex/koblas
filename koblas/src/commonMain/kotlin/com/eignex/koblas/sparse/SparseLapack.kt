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
interface SparseLapack : Backend {
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
    fun factor(a: SparseMatrix, equilibrate: Boolean = false, dropTolerance: Double = NO_DROP): SparseFactorization =
        SparseLu.factorCsc(a, equilibrate, dropTolerance)
}
