package com.eignex.koblas.sparse.basis

import com.eignex.koblas.Backend
import com.eignex.koblas.core.F64SparseMatrix

/**
 * Basis solvers as a backend half of its own, separate from the sparse LU beside it.
 *
 * They are separate because the two contracts want different libraries. A basis solver names its entering
 * column by index into a matrix fixed for its lifetime, which is what lets a backend keep the factors where
 * the columns lie and solve hypersparsely.
 * [com.eignex.koblas.sparse.F64BasisFactorization] takes any column at all, so a backend carrying the basis
 * independently of a matrix answers it and one reading its columns out of a matrix by index cannot: the
 * update would go through, but the refactorization behind it would have no column to rebuild from.
 *
 * Held in one seam these would share a winner, and the strongest at one contract would take the other from
 * whoever was strongest there. Held apart, each goes to the backend that has it.
 */
public interface F64BasisSolvers : Backend {
    /**
     * A solver for bases drawn from the columns of [a], which outlives every basis taken from it.
     *
     * [a] carries the logical columns explicitly, so a basis slot naming one is an ordinary column rather
     * than a case for the solver to know about, and it must have at least as many columns as rows.
     */
    public fun basisSolver(a: F64SparseMatrix): F64BasisSolver
}
