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
     */
    fun factor(a: SparseMatrix, equilibrate: Boolean = false): SparseFactorization = SparseLu.factorCsc(a, equilibrate)
}
