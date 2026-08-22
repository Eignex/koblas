package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.factorization.ldl.*
import com.eignex.koblas.sparse.factorization.lu.*
import com.eignex.koblas.sparse.symbolic.*

/** Sparse factorizations as a backend half. */
public interface F64SparseLapack : Backend {
    /**
     * Factorize the square [a] into something solvable. A singular matrix comes back as a factorization
     * reporting `singular` rather than as an exception, with a `failedAt` counting elimination steps rather
     * than naming a column: the step that fails is the one with no acceptable pivot left, so there is no
     * column of [a] to attribute it to.
     *
     * @param a the square matrix to factorize.
     * @param equilibrate scale rows by a power of two first; the solves and the determinant undo it.
     * @param dropTolerance discard produced entries this far below the largest magnitude, giving an
     *   incomplete factorization.
     */
    public fun factor(
        a: F64SparseMatrix,
        equilibrate: Boolean = false,
        dropTolerance: Double = NO_DROP,
    ): F64SparseFactorization

    /**
     * Analyse a symmetric [a]'s pattern, the elimination tree and the nonzero pattern of L, reading no
     * values. Reusable, so every matrix with this pattern factors against one analysis.
     */
    public fun analyze(a: F64SparseMatrix, ordering: SparseOrdering = SparseOrdering.MinimumDegree): SparseSymbolic

    /**
     * `A = L·D·Lᵀ` for a symmetric [a], analysing the pattern first. Accepts an indefinite matrix by
     * default, and under that default an exact zero pivot is singular and reports the column of [a] it
     * stopped at, which the fill-reducing ordering makes distinct from the step it stopped on.
     */
    public fun ldl(
        a: F64SparseMatrix,
        policy: SparseLdlPolicy = SparseLdlPolicy.Indefinite,
        ordering: SparseOrdering = SparseOrdering.MinimumDegree,
    ): F64SparseFactorization

    /**
     * The Cholesky factorization of a symmetric positive-definite [a], as `L·D·Lᵀ` with every pivot
     * positive. The classical factor `L·√D` is available through `F64SparseLdl.choleskyFactor`.
     *
     * @throws com.eignex.koblas.NotPositiveDefinite on a non-positive pivot, naming that pivot's column
     *   of [a].
     */
    public fun cholesky(
        a: F64SparseMatrix,
        policy: SparseLdlPolicy = SparseLdlPolicy.Strict,
        ordering: SparseOrdering = SparseOrdering.MinimumDegree,
    ): F64SparseFactorization = ldl(a, policy, ordering)

    /**
     * Solve `A·x = b` from [f] into [out], `Aᵀ·x = b` when [transpose]. The work belongs to the
     * factorization; this is here so the seam reads the same from the sparse side as [com.eignex.koblas
     * .dense.F64Lapack.solveInto] does from the dense one.
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
