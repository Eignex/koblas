package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64ProductFormBasisSolver
import com.eignex.koblas.sparse.factorization.lu.NO_DROP

/** Sparse LU factorization as a backend half. */
public interface F64SparseLu : Backend {
    /**
     * Factorize the square [a] into something solvable. A singular matrix comes back as a factorization
     * reporting `singular` rather than as an exception, with a `failedAt` counting elimination steps rather
     * than naming a column: the step that fails is the one with no acceptable pivot left, so there is no
     * column of [a] to attribute it to.
     *
     * @param a the square matrix to factorize.
     * @param equilibrate scale rows by a power of two first; the solves undo it.
     * @param dropTolerance discard produced entries this far below the largest magnitude, giving an
     *   incomplete factorization.
     */
    public fun factor(
        a: F64SparseMatrix,
        equilibrate: Boolean = false,
        dropTolerance: Double = NO_DROP,
    ): F64SparseFactorization

    /**
     * Factor [a], reusing compatible state from [previous] when this backend can. The returned
     * factorization supersedes [previous], which must not be solved after this call. Backends that cannot
     * reuse it answer as [factor] would.
     */
    public fun refactor(
        previous: F64SparseFactorization,
        a: F64SparseMatrix,
        equilibrate: Boolean = false,
        dropTolerance: Double = NO_DROP,
    ): F64SparseFactorization = factor(a, equilibrate, dropTolerance)

    /**
     * Whether [factorBasis] answers with a factorization that updates its factors in place. When false a
     * replacement costs a factorization, so a caller pacing its own refactorizations has nothing left to pace.
     */
    public val supportsBasisUpdates: Boolean get() = false

    /**
     * Factor a simplex [basis] for column replacements.
     *
     * A general sparse LU backend need not support factor updates. The default refactorizes on every
     * replacement, which every backend can do; [supportsBasisUpdates] tells the two apart.
     */
    public fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization {
        requireSquare(basis, "factorBasis")
        return F64RefactoringBasisFactorization(this, basis, factor(basis))
    }

    /**
     * A solver for bases drawn from the columns of [a], for a simplex that pivots rather than a caller
     * factoring one matrix at a time.
     *
     * The portable answer keeps a factorization of the basis and folds each pivot in as an elementary
     * transform. A backend whose library updates its own factors overrides this with that, which is where
     * the seam pays: the basis is named by index into [a], so neither side assembles a square matrix per
     * refactorization.
     */
    public fun basisSolver(a: F64SparseMatrix): F64BasisSolver = F64ProductFormBasisSolver(a, this)

    /**
     * Solve `A·x = b` from [f] into [out], `Aᵀ·x = b` when [transpose]. The work belongs to the
     * factorization; this is here so the seam reads the same from the sparse side as [com.eignex.koblas
     * .dense.F64Decompositions.solveInto] does from the dense one.
     */
    public fun solveInto(
        f: F64SparseFactorization,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray = f.solveInto(b, out, transpose, workspace)

    /** [solveInto] into a fresh vector. */
    public fun solve(f: F64SparseFactorization, b: DoubleArray, transpose: Boolean = false): DoubleArray =
        f.solve(b, transpose)
}
