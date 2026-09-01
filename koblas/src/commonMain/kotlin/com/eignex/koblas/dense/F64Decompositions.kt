@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix

/** Dense factorizations as a backend half. */
public interface F64Decompositions : Backend {

    /** The vector kernels this half's inherited routines run on; the installed ones by default. */
    public val kernels: F64Kernels get() = koblas.kernels

    /** LU factorization with partial pivoting of any `m×n` [a] (LAPACK `dgetrf`). [a] is not modified. */
    public fun factor(a: F64DenseMatrix): F64LuDecomposition

    /** Refactorize [a] into [out]'s existing buffers, returning [out]. [out] must have [a]'s shape and
     *  its previous contents are discarded. */
    public fun factorInto(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
        requireShape(out.rows == a.rows && out.cols == a.cols) {
            "factorInto: out is ${out.rows}x${out.cols}, expected ${a.rows}x${a.cols}"
        }
        val fresh = factor(a)
        fresh.lu.copyInto(out.lu)
        fresh.piv.copyInto(out.piv)
        out.failedAt = fresh.failedAt
        return out
    }

    /** Solve `A · x = b`, or `Aᵀ · x = b` when [transpose], for a square factorization [lu] (LAPACK `dgetrs`).
     *  @throws com.eignex.koblas.DimensionMismatch if [lu] is rectangular. */
    public fun solve(lu: F64LuDecomposition, b: DoubleArray, transpose: Boolean = false): DoubleArray =
        solveInto(lu, b, DoubleArray(lu.rows), transpose)

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
     *  against a square factorization [lu] (LAPACK `dgetrs` with `nrhs`).
     *  @throws com.eignex.koblas.DimensionMismatch if [lu] is rectangular. */
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

    /**
     * Bunch-Kaufman numerically pivoted symmetric-indefinite factorization (LAPACK `dsytrf`, lower).
     *
     * It reads only [a]'s lower triangle. This is deliberately distinct from sparse
     * quasi-definite LDL: its pivots are selected from the numeric values for stability, not only from a
     * fill-reducing ordering.
     */
    public fun pivotedSymmetricIndefinite(
        a: F64DenseMatrix,
        workspace: Workspace? = null,
    ): F64PivotedSymmetricIndefiniteDecomposition

    /** Solve `A · x = b` for [factor] (LAPACK `dsytrs`). */
    public fun solve(factor: F64PivotedSymmetricIndefiniteDecomposition, b: DoubleArray): DoubleArray =
        solveInto(factor, b, DoubleArray(factor.n))

    /** Solve `A · x = b` into [out], which is returned. [out] may be [b]. */
    public fun solveInto(
        factor: F64PivotedSymmetricIndefiniteDecomposition,
        b: DoubleArray,
        out: DoubleArray,
    ): DoubleArray

    /** Solve `A · X = B` for all right-hand-side columns of [b] at once against a symmetric indefinite
     *  factorization (LAPACK `dsytrs` with `nrhs`). */
    public fun solve(factor: F64PivotedSymmetricIndefiniteDecomposition, b: F64DenseMatrix): F64DenseMatrix = solveInto(
        factor,
        b,
        F64DenseMatrix(b.rows, b.cols),
    )

    /** Solve `A · X = B` into [out], which is returned. [out] may be [b]. */
    public fun solveInto(
        factor: F64PivotedSymmetricIndefiniteDecomposition,
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
    public fun solve(qr: F64PivotedQrDecomposition, b: DoubleArray, workspace: Workspace? = null): DoubleArray =
        solveInto(qr, b, DoubleArray(qr.n), workspace)

    /** [solve] into [out], which is returned. */
    public fun solveInto(
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

    /** Solve from a QR factorization. By default, finds the least-squares solution `min ‖A·x − b‖₂` for a tall
     *  or square `A`; it requires full column rank and returns `R⁻¹·(Qᵀb)`. With [minimumNorm], finds the
     *  minimum-norm solution of a consistent wide system from `qr(Aᵀ)`; it requires full row rank. */
    public fun solve(
        qr: F64QrDecomposition,
        b: DoubleArray,
        minimumNorm: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray = solveInto(qr, b, DoubleArray(if (minimumNorm) qr.m else qr.n), minimumNorm, workspace)

    /** [solve] into [out], which is returned. Its length is `n` by default and `m` with [minimumNorm]. A
     *  [workspace] lends the intermediate for applying `Q` or `Qᵀ`. */
    public fun solveInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        minimumNorm: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray

    /** Order-of-magnitude estimate of `1 / (anorm · est(‖A⁻¹‖₁))` (LAPACK `dgecon`) for a square factor, where
     *  [anorm] is the 1-norm of the unfactored matrix (see [norm1]). Returns `1.0` when `n == 0` and `0.0`
     *  when singular.
     *  @throws com.eignex.koblas.DimensionMismatch if [lu] is rectangular. */
    public fun rcond(lu: F64LuDecomposition, anorm: Double, workspace: Workspace? = null): Double

    /** Cholesky factorization `A = L·Lᵀ` of a symmetric positive-definite [a] (`dpotrf` with `uplo = 'L'`).
     *  Reads only the lower triangle of [a], so an upper-only matrix factors to silent nonsense.
     *  @throws com.eignex.koblas.NotPositiveDefinite at the first non-positive pivot unless [policy] allows it.
     */
    public fun cholesky(a: F64DenseMatrix, policy: CholeskyPolicy = CholeskyPolicy.Strict): F64CholeskyDecomposition

    /** Solve `A · x = b` for [chol] into a fresh vector (LAPACK `dpotrs`). */
    public fun solve(chol: F64CholeskyDecomposition, b: DoubleArray): DoubleArray =
        solveInto(chol, b, DoubleArray(chol.n))

    /** Solve `A · x = b` for [chol] into [out], which is returned. [out] may be [b]. */
    public fun solveInto(chol: F64CholeskyDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray {
        val n = chol.n
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        if (out !== b) b.copyInto(out)
        val ld = chol.l.data
        trsvCore(kernels, ld, n, out, lower = true, transpose = false, unitDiag = false)
        trsvCore(kernels, ld, n, out, lower = true, transpose = true, unitDiag = false)
        return out
    }

    /** Solve `A · X = B` for every right-hand-side column of [b] into a fresh dense result. */
    public fun solve(chol: F64CholeskyDecomposition, b: F64DenseMatrix): F64DenseMatrix =
        solveInto(chol, b, F64DenseMatrix(b.rows, b.cols))

    /** Solve `A · X = B` into [out], which is returned. [out] may be [b]; zero RHS columns return unchanged.
     *  [workspace] provides independent per-column staging buffers. */
    public fun solveInto(
        chol: F64CholeskyDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        workspace: Workspace? = null,
    ): F64DenseMatrix {
        requireSolveShapes(chol.n, b, out)
        return solveColumnwise(b, out, chol.n, b.cols, workspace) { column, destination ->
            solveInto(chol, column, destination)
        }
    }

    /** Invert a square matrix from its LU factorization, returning `A⁻¹` given `P·A = L·U` (LAPACK `dgetri`).
     *  Prefer [solve] to apply `A⁻¹`, which costs less and is more accurate.
     *  @throws com.eignex.koblas.SingularMatrix if [lu] is singular; the position is [F64LuDecomposition.failedAt].
     *  @throws com.eignex.koblas.DimensionMismatch if [lu] is rectangular.
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

/** Sweep cap for the [F64Decompositions.rcond] estimator. */
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
    // Borrowed rather than taken because [solveOne] is the caller's, and a native solve in it reports a
    // failure by throwing.
    workspace.borrow(n) { col ->
        workspace.borrow(n) { solved ->
            for (c in 0 until nrhs) {
                b.data.copyInto(col, 0, c * n, (c + 1) * n)
                val x = solveOne(col, solved)
                x.copyInto(out.data, c * n, 0, n)
            }
        }
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
