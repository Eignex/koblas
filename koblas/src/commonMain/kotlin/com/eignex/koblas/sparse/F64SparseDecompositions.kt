package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix

/**
 * Sparse factorizations as a backend half, the counterpart of the dense [com.eignex.koblas.dense
 * .F64Decompositions]: one seam carrying every factorization of a sparse matrix rather than one seam per
 * kind, so a library offering two of them fills one half instead of competing with itself for two.
 *
 * Only what a factorization of that kind universally does is here. The libraries behind this seam are
 * specialised rather than interchangeable, and each carries state the others do not: KLU reuses a symbolic
 * analysis across matrices of one pattern, BASICLU keeps a basis through a sequence of column replacements,
 * HFactor holds factors over a matrix that outlives them. Each of those had exactly one implementer when it
 * sat here, which makes it a concrete routine on the backend that has it rather than a method every other
 * backend inherits and declines. What a factorization is allowed to do to the matrix on the way in — row
 * equilibration, a drop tolerance — is policy for the backend's own constructor, beside the settings that
 * say how.
 *
 * A caller wanting one of those reaches the backend by name through
 * [com.eignex.koblas.backendNamed] with the capability it fills, rather than through the seam.
 */
public interface F64SparseDecompositions : Backend {
    /**
     * Factorize the square [a] into something solvable. A singular matrix comes back as a factorization
     * reporting `singular` rather than as an exception, with a `failedAt` counting elimination steps rather
     * than naming a column: the step that fails is the one with no acceptable pivot left, so there is no
     * column of [a] to attribute it to.
     */
    public fun factor(a: F64SparseMatrix): F64SparseLuFactorization

    /**
     * Cholesky factorization `A = L·Lᵀ` of a symmetric positive-definite [a]. Only the lower triangle is
     * read, as the dense [com.eignex.koblas.dense.F64Decompositions.cholesky] does, so anything stored above
     * the diagonal is ignored rather than checked against its mirror.
     *
     * Unlike [factor] this raises rather than reports. A missing LU pivot means the matrix is singular, which
     * the factorization can carry as a state; a non-positive Cholesky pivot means the matrix was not what the
     * caller said it was, and a factorization of a matrix that was never given is not a state worth handing
     * back.
     *
     * @throws com.eignex.koblas.NotPositiveDefinite at the first column whose pivot is not positive.
     */
    public fun cholesky(a: F64SparseMatrix): F64SparseCholeskyFactorization

    /**
     * Factorization `A = L·D·Lᵀ` of a symmetric [a], with `L` unit lower triangular. Only the lower triangle
     * is read, as [cholesky] does.
     *
     * Where [cholesky] takes a square root and so demands a positive pivot, this does not, which is what it
     * is for: an indefinite matrix has an `L·D·Lᵀ` and has no `L·Lᵀ`. A zero pivot comes back as a
     * factorization reporting `singular` at that column, the way [factor] reports one, rather than raising
     * the way [cholesky] does; a negative pivot is not a failure at all.
     *
     * Unlike dense [com.eignex.koblas.dense.F64Decompositions.pivotedSymmetricIndefinite], neither this nor
     * any library behind this seam selects numerical pivots: the permutation is chosen to limit fill and
     * nothing reorders on the numbers. So this is the factorization for a matrix that is quasi-definite,
     * which is what an interior point method's KKT system is, and it can be arbitrarily ill conditioned on a
     * general indefinite one. A caller that cannot promise quasi-definiteness wants [factor], whose pivoting
     * is numerical.
     */
    public fun quasiDefiniteLdl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization

    /**
     * @deprecated Sparse LDL is quasi-definite and numerically unpivoted. Use [quasiDefiniteLdl].
     */
    @Deprecated(
        "Sparse LDL is quasi-definite and numerically unpivoted; use quasiDefiniteLdl.",
        ReplaceWith("quasiDefiniteLdl(a)"),
    )
    public fun ldl(a: F64SparseMatrix): F64QuasiDefiniteLdlFactorization = quasiDefiniteLdl(a)

    /**
     * QR factorization of a tall or square [a], for the least-squares solve `min ‖A·x − b‖₂`. Its factor is
     * an [F64SparseQrFactorization] rather than an [F64SparseFactorization], which is square.
     *
     * @throws IllegalArgumentException if [a] has fewer rows than columns.
     */
    public fun qr(a: F64SparseMatrix): F64SparseQrFactorization

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

    /** Solve `A · X = B` from [f] into a fresh dense result. */
    public fun solve(
        f: F64SparseFactorization,
        b: F64DenseMatrix,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): F64DenseMatrix = f.solveInto(
        b,
        F64DenseMatrix(if (transpose) f.n else f.n, b.cols),
        transpose,
        workspace,
    )

    /** Solve `A · X = B` from [f] into [out], which is returned. [out] may be [b]. */
    public fun solveInto(
        f: F64SparseFactorization,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): F64DenseMatrix = f.solveInto(b, out, transpose, workspace)
}
