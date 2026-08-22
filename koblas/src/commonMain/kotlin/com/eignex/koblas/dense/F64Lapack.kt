@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.norm1
import com.eignex.koblas.requireShape
import com.eignex.koblas.transpose

/** Dense factorizations as a backend half. */
public interface F64Lapack : Backend {

    /** The vector kernels this half's inherited routines run on; the installed ones by default. */
    public val vectorKernels: F64VectorKernels get() = koblas.vectorKernels

    /** LU factorization with partial pivoting of a square [a] (LAPACK `dgetrf`). [a] is not modified. */
    public fun factor(a: F64DenseMatrix): F64LuDecomposition

    /** Refactorize [a] into [out]'s existing buffers, returning [out]. [out] must have [a]'s dimension and
     *  its previous contents are discarded. */
    public fun factorInto(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        val fresh = factor(a)
        fresh.lu.copyInto(out.lu)
        fresh.piv.copyInto(out.piv)
        out.failedAt = fresh.failedAt
        return out
    }

    /** Solve `A · x = b`, or `Aᵀ · x = b` when [transpose], for the factorization [lu] (LAPACK `dgetrs`). */
    public fun solve(lu: F64LuDecomposition, b: DoubleArray, transpose: Boolean = false): DoubleArray =
        solveInto(lu, b, DoubleArray(lu.n), transpose)

    /** Solve `A · x = b`, or `Aᵀ · x = b` when [transpose], into [out], which is returned. [out] may be [b],
     *  and a [workspace] lends the transposed direction's staging buffer. */
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    public fun solveInto(
        lu: F64LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray

    /** Solve `A · X = B`, or `Aᵀ · X = B` when [transpose], for all right-hand-side columns of [b] at once
     *  (LAPACK `dgetrs` with `nrhs`). */
    public fun solve(lu: F64LuDecomposition, b: F64DenseMatrix, transpose: Boolean = false): F64DenseMatrix =
        solveInto(lu, b, F64DenseMatrix(b.rows, b.cols), transpose)

    /** Solve `A · X = B`, or `Aᵀ · X = B` when [transpose], into [out], which is returned. [out] may be [b],
     *  and a [workspace] lends the transposed direction's `n·nrhs` staging block. */
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    public fun solveInto(
        lu: F64LuDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): F64DenseMatrix

    /** Symmetric indefinite factorization `A = L·D·Lᵀ` with Bunch-Kaufman pivoting (LAPACK `dsytrf`, lower).
     *  Reads only the lower triangle of [a], so an upper-only matrix factors to silent nonsense. */
    public fun ldl(a: F64DenseMatrix, workspace: Workspace? = null): F64LdlDecomposition

    /** Solve `A · x = b` for a symmetric indefinite factorization [ldl] (LAPACK `dsytrs`). */
    public fun solve(ldl: F64LdlDecomposition, b: DoubleArray): DoubleArray = solveInto(ldl, b, DoubleArray(ldl.n))

    /** Solve `A · x = b` into [out], which is returned. [out] may be [b]. */
    public fun solveInto(ldl: F64LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray

    /** Solve `A · X = B` for all right-hand-side columns of [b] at once against a symmetric indefinite
     *  factorization (LAPACK `dsytrs` with `nrhs`). */
    public fun solve(ldl: F64LdlDecomposition, b: F64DenseMatrix): F64DenseMatrix = solveInto(
        ldl,
        b,
        F64DenseMatrix(b.rows, b.cols),
    )

    /** Solve `A · X = B` into [out], which is returned. [out] may be [b]. */
    public fun solveInto(
        ldl: F64LdlDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        workspace: Workspace? = null,
    ): F64DenseMatrix

    /** QR factorization `A = Q·R` of an `m×n` [a] via Householder reflections (LAPACK `dgeqrf`). [a] is not
     *  modified, any shape is accepted, and rank deficiency is not detected. */
    public fun qr(a: F64DenseMatrix, workspace: Workspace? = null): F64QrDecomposition

    /** QR with column pivoting, `A·P = Q·R` (LAPACK `dgeqp3`), reporting [F64PivotedQrDecomposition.rank] as the
     *  count of leading diagonal entries with `|R_kk| > tolerance · |R₀₀|`. [tolerance] is a fraction of
     *  `|R₀₀|`; [AUTOMATIC_RANK_TOLERANCE] derives one from the shape, `max(m, n) · ε`, and a negative value
     *  is rejected.
     *  @throws IllegalArgumentException if [tolerance] is negative. */
    public fun qrPivoted(
        a: F64DenseMatrix,
        tolerance: Double = AUTOMATIC_RANK_TOLERANCE,
        workspace: Workspace? = null,
    ): F64PivotedQrDecomposition

    /** Least-squares solve from a pivoted factorization, with the column permutation undone. A rank-deficient
     *  factorization returns the basic solution, not the minimum-norm one: zero outside the pivoted rank. */
    public fun solveLeastSquares(
        qr: F64PivotedQrDecomposition,
        b: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray = solveLeastSquaresInto(qr, b, DoubleArray(qr.n), workspace)

    /** [solveLeastSquares] into [out], which is returned. */
    public fun solveLeastSquaresInto(
        qr: F64PivotedQrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray

    /** Apply `Q`, or `Qᵀ` when [transpose], from [qr] to a length-`m` [y], into a fresh result
     *  (LAPACK `dormqr` restricted to a single column). */
    public fun applyQ(qr: F64QrDecomposition, y: DoubleArray, transpose: Boolean = false): DoubleArray =
        applyQInto(qr, y, DoubleArray(qr.m), transpose)

    /** Apply `Q`, or `Qᵀ` when [transpose], from [qr] to [y] into [out], which is returned. [out] may
     *  be [y]. */
    public fun applyQInto(
        qr: F64QrDecomposition,
        y: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
    ): DoubleArray

    /** Least-squares solve `min ‖A·x − b‖₂` from the factorization (the `dgels` shape). Requires `m ≥ n` and
     *  full column rank, and returns the length-`n` solution `R⁻¹·(Qᵀb)`. */
    public fun solveLeastSquares(qr: F64QrDecomposition, b: DoubleArray, workspace: Workspace? = null): DoubleArray =
        solveLeastSquaresInto(qr, b, DoubleArray(qr.n), workspace)

    /** Least-squares solve into [out], which has length `n` and is returned. A [workspace] lends the
     *  length-`m` intermediate for `Qᵀb`. */
    public fun solveLeastSquaresInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray

    /** Minimum-norm solution of the consistent underdetermined system `A · x = b` for a wide `m×n` A of full
     *  row rank. Pass the factorization `qr(Aᵀ)`; rank deficiency is not detected. */
    public fun solveMinimumNorm(qr: F64QrDecomposition, b: DoubleArray, workspace: Workspace? = null): DoubleArray =
        solveMinimumNormInto(qr, b, DoubleArray(qr.m), workspace)

    /** Minimum-norm solve into [out], which has length `m` (the wide system's column count) and is returned.
     *  A [workspace] lends the length-`n` intermediate. */
    public fun solveMinimumNormInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray

    /** Order-of-magnitude estimate of `1 / (anorm · est(‖A⁻¹‖₁))` (LAPACK `dgecon`), where [anorm] is the
     *  1-norm of the unfactored matrix (see [norm1]). Returns `1.0` when `n == 0` and `0.0` when singular. */
    public fun rcond(lu: F64LuDecomposition, anorm: Double, workspace: Workspace? = null): Double

    /** Cholesky factorization `A = L·Lᵀ` of a symmetric positive-definite [a] (`dpotrf` with `uplo = 'L'`).
     *  Reads only the lower triangle of [a], so an upper-only matrix factors to silent nonsense.
     *  @throws com.eignex.koblas.NotPositiveDefinite at the first non-positive pivot unless [policy] allows it.
     */
    public fun cholesky(a: F64DenseMatrix, policy: CholeskyPolicy = CholeskyPolicy.Strict): F64CholeskyDecomposition

    /** Solve `A · x = b` for the Cholesky factorization [chol] (LAPACK `dpotrs`). [b] is not modified. */
    public fun solve(chol: F64CholeskyDecomposition, b: DoubleArray): DoubleArray {
        val n = chol.n
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        val x = b.copyOf()
        val ld = chol.l.data
        trsvCore(vectorKernels, ld, n, x, lower = true, transpose = false, unitDiag = false)
        trsvCore(vectorKernels, ld, n, x, lower = true, transpose = true, unitDiag = false)
        return x
    }

    /** Invert a general matrix from its LU factorization, returning `A⁻¹` given `P·A = L·U` (LAPACK `dgetri`).
     *  Prefer [solve] to apply `A⁻¹`, which costs less and is more accurate.
     *  @throws com.eignex.koblas.SingularMatrix if [lu] is singular; the position is [F64LuDecomposition.failedAt].
     */
    public fun invert(lu: F64LuDecomposition, workspace: Workspace? = null): F64DenseMatrix

    /** Invert a triangular matrix into a fresh result (LAPACK `dtrtri`), returning `T⁻¹` for the [lower] or
     *  upper triangle of the square [a], taking the diagonal as 1 when [unitDiag].
     *  @throws com.eignex.koblas.SingularMatrix naming the first zero diagonal position.
     */
    public fun trtri(a: F64DenseMatrix, lower: Boolean, unitDiag: Boolean = false): F64DenseMatrix

    /** Invert an SPD matrix from its Cholesky factorization, returning `A⁻¹` given [chol] (LAPACK `dpotri`). */
    public fun invert(chol: F64CholeskyDecomposition, workspace: Workspace? = null): F64DenseMatrix
}

/** Sweep cap for the [F64Lapack.rcond] estimator. */
internal const val RCOND_MAX_SWEEPS = 5

/** The shapes a multi-right-hand-side solve needs: [b] with [n] rows, and [out] matching it column for column. */
internal fun requireSolveShapes(n: Int, b: F64DenseMatrix, out: F64DenseMatrix) {
    requireShape(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
    requireShape(out.rows == n && out.cols == b.cols) {
        "solve: out is ${out.rows}x${out.cols}, expected ${n}x${b.cols}"
    }
}

/** Run [solveOne] over each column of [b], reassembling the results into an `n × nrhs` matrix. */
@Suppress("LongParameterList") // shape, destination and scratch alongside the per-column solve
internal inline fun solveColumnwise(
    b: F64DenseMatrix,
    out: F64DenseMatrix,
    n: Int,
    nrhs: Int,
    workspace: Workspace?,
    solveOne: (column: DoubleArray, destination: DoubleArray) -> DoubleArray,
): F64DenseMatrix {
    if (nrhs == 0) return out
    val col = workspace?.take(n) ?: DoubleArray(n)
    val solved = workspace?.take(n) ?: DoubleArray(n)
    for (c in 0 until nrhs) {
        b.data.copyInto(col, 0, c * n, (c + 1) * n)
        val x = solveOne(col, solved)
        x.copyInto(out.data, c * n, 0, n)
    }
    if (workspace != null) {
        workspace.release(solved)
        workspace.release(col)
    }
    return out
}

/**
 * Move rows between pivot order and original order for an `nrhs`-column right-hand side, the permutation
 * an LU solve applies before its triangular sweeps and undoes after them.
 *
 * [gather] reads `src` in original order and writes pivot order; otherwise the reverse.
 */
internal fun permuteRows(src: DoubleArray, dst: DoubleArray, n: Int, nrhs: Int, piv: IntArray, gather: Boolean) {
    for (c in 0 until nrhs) {
        val base = c * n
        if (gather) {
            for (i in 0 until n) dst[base + i] = src[base + piv[i]]
        } else {
            for (i in 0 until n) dst[base + piv[i]] = src[base + i]
        }
    }
}
