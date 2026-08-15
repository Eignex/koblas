@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.norm1
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import com.eignex.koblas.transpose
import kotlin.math.abs
import kotlin.math.sqrt

/** Dense factorizations as a backend half. */
public interface Lapack : Backend {

    /** The vector kernels this half's inherited routines run on; the installed ones by default. */
    public val vectorKernels: VectorKernels get() = koblas.vectorKernels

    /** LU factorization with partial pivoting of a square [a] (LAPACK `dgetrf`). [a] is not modified. */
    public fun factor(a: DenseMatrix): LuDecomposition

    /** Refactorize [a] into [out]'s existing buffers, returning [out]. [out] must have [a]'s dimension and
     *  its previous contents are discarded. */
    public fun factorInto(a: DenseMatrix, out: LuDecomposition): LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        val fresh = factor(a)
        fresh.lu.copyInto(out.lu)
        fresh.piv.copyInto(out.piv)
        out.failedAt = fresh.failedAt
        return out
    }

    /** Solve `A · x = b`, or `Aᵀ · x = b` when [transpose], for the factorization [lu] (LAPACK `dgetrs`). */
    public fun solve(lu: LuDecomposition, b: DoubleArray, transpose: Boolean = false): DoubleArray =
        solveInto(lu, b, DoubleArray(lu.n), transpose)

    /** Solve `A · x = b`, or `Aᵀ · x = b` when [transpose], into [out], which is returned. [out] may be [b],
     *  and a [workspace] lends the transposed direction's staging buffer. */
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    public fun solveInto(
        lu: LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray

    /** Solve `A · X = B`, or `Aᵀ · X = B` when [transpose], for all right-hand-side columns of [b] at once
     *  (LAPACK `dgetrs` with `nrhs`). */
    public fun solve(lu: LuDecomposition, b: DenseMatrix, transpose: Boolean = false): DenseMatrix =
        solveInto(lu, b, DenseMatrix(b.rows, b.cols), transpose)

    /** Solve `A · X = B`, or `Aᵀ · X = B` when [transpose], into [out], which is returned. [out] may be [b],
     *  and a [workspace] lends the transposed direction's `n·nrhs` staging block. */
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    public fun solveInto(
        lu: LuDecomposition,
        b: DenseMatrix,
        out: DenseMatrix,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DenseMatrix {
        requireFactored(lu.failedAt, "solve")
        val n = lu.n
        val nrhs = b.cols
        requireShape(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        requireShape(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        if (nrhs == 0) return out
        val f = lu.lu
        return if (transpose) {
            val y = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
            b.data.copyInto(y)
            trsmCore(vectorKernels, f, n, y, nrhs, lower = false, transpose = true, unitDiag = false)
            trsmCore(vectorKernels, f, n, y, nrhs, lower = true, transpose = true, unitDiag = true)
            scatterRows(y, out.data, n, nrhs, lu.piv)
            workspace?.release(y)
            out
        } else {
            // The gather cannot read B in place once out aliases it, so that case stages.
            if (out === b) {
                val staged = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
                gatherRows(b.data, staged, n, nrhs, lu.piv)
                staged.copyInto(out.data)
                workspace?.release(staged)
            } else {
                gatherRows(b.data, out.data, n, nrhs, lu.piv)
            }
            trsmCore(vectorKernels, f, n, out.data, nrhs, lower = true, transpose = false, unitDiag = true)
            trsmCore(vectorKernels, f, n, out.data, nrhs, lower = false, transpose = false, unitDiag = false)
            out
        }
    }

    /** Symmetric indefinite factorization `A = L·D·Lᵀ` with Bunch-Kaufman pivoting (LAPACK `dsytrf`, lower).
     *  Reads only the lower triangle of [a], so an upper-only matrix factors to silent nonsense. */
    public fun ldl(a: DenseMatrix, workspace: Workspace? = null): LdlDecomposition

    /** Solve `A · x = b` for a symmetric indefinite factorization [ldl] (LAPACK `dsytrs`). */
    public fun solve(ldl: LdlDecomposition, b: DoubleArray): DoubleArray = solveInto(ldl, b, DoubleArray(ldl.n))

    /** Solve `A · x = b` into [out], which is returned. [out] may be [b]. */
    public fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray

    /** Solve `A · X = B` for all right-hand-side columns of [b] at once against a symmetric indefinite
     *  factorization (LAPACK `dsytrs` with `nrhs`). */
    public fun solve(ldl: LdlDecomposition, b: DenseMatrix): DenseMatrix = solveInto(
        ldl,
        b,
        DenseMatrix(b.rows, b.cols),
    )

    /** Solve `A · X = B` into [out], which is returned. [out] may be [b]. */
    public fun solveInto(
        ldl: LdlDecomposition,
        b: DenseMatrix,
        out: DenseMatrix,
        workspace: Workspace? = null,
    ): DenseMatrix {
        requireFactored(ldl.failedAt, "solve")
        val n = ldl.n
        val nrhs = b.cols
        requireShape(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        requireShape(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        return solveColumnwise(b, out, n, nrhs, workspace) { col, dst -> solveInto(ldl, col, dst) }
    }

    /** QR factorization `A = Q·R` of an `m×n` [a] via Householder reflections (LAPACK `dgeqrf`). [a] is not
     *  modified, any shape is accepted, and rank deficiency is not detected. */
    public fun qr(a: DenseMatrix, workspace: Workspace? = null): QrDecomposition

    /** QR with column pivoting, `A·P = Q·R` (LAPACK `dgeqp3`), reporting [PivotedQrDecomposition.rank] as the
     *  count of leading diagonal entries with `|R_kk| > tolerance · |R₀₀|`. [tolerance] defaults to `max(m, n) · ε`. */
    public fun qrPivoted(
        a: DenseMatrix,
        tolerance: Double = AUTOMATIC_RANK_TOLERANCE,
        workspace: Workspace? = null,
    ): PivotedQrDecomposition = ReferenceLapack(vectorKernels).qrPivoted(a, tolerance, workspace)

    /** Least-squares solve from a pivoted factorization, with the column permutation undone. A rank-deficient
     *  factorization returns the basic solution, not the minimum-norm one: zero outside the pivoted rank. */
    public fun solveLeastSquares(
        qr: PivotedQrDecomposition,
        b: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray = solveLeastSquaresInto(qr, b, DoubleArray(qr.n), workspace)

    /** [solveLeastSquares] into [out], which is returned. */
    public fun solveLeastSquaresInto(
        qr: PivotedQrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray {
        requireShape(b.size == qr.m) { "solveLeastSquares: b length ${b.size} != ${qr.m}" }
        requireShape(out.size == qr.n) { "solveLeastSquares: out length ${out.size} != ${qr.n}" }
        val rank = qr.rank
        val y = workspace?.take(qr.m) ?: DoubleArray(qr.m)
        applyQInto(qr.factorization, b, y, transpose = true)
        trsvCore(
            vectorKernels,
            qr.factorization.qr,
            rank,
            y,
            lda = qr.m,
            lower = false,
            transpose = false,
            unitDiag = false,
        )
        out.fill(0.0)
        for (k in 0 until rank) out[qr.pivots[k]] = y[k]
        workspace?.release(y)
        return out
    }

    /** Apply `Q`, or `Qᵀ` when [transpose], from [qr] to a length-`m` [y], into a fresh result
     *  (LAPACK `dormqr` restricted to a single column). */
    public fun applyQ(qr: QrDecomposition, y: DoubleArray, transpose: Boolean = false): DoubleArray =
        applyQInto(qr, y, DoubleArray(qr.m), transpose)

    /** Apply `Q`, or `Qᵀ` when [transpose], from [qr] to [y] into [out], which is returned. [out] may
     *  be [y]. */
    public fun applyQInto(
        qr: QrDecomposition,
        y: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
    ): DoubleArray

    /** Least-squares solve `min ‖A·x − b‖₂` from the factorization (the `dgels` shape). Requires `m ≥ n` and
     *  full column rank, and returns the length-`n` solution `R⁻¹·(Qᵀb)`. */
    public fun solveLeastSquares(qr: QrDecomposition, b: DoubleArray, workspace: Workspace? = null): DoubleArray =
        solveLeastSquaresInto(qr, b, DoubleArray(qr.n), workspace)

    /** Least-squares solve into [out], which has length `n` and is returned. A [workspace] lends the
     *  length-`m` intermediate for `Qᵀb`. */
    public fun solveLeastSquaresInto(
        qr: QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray {
        requireShape(qr.m >= qr.n) { "solveLeastSquares requires m >= n, got ${qr.m}x${qr.n}" }
        requireShape(b.size == qr.m) { "solveLeastSquares: b length ${b.size} != ${qr.m}" }
        requireShape(out.size == qr.n) { "solveLeastSquares: out length ${out.size} != ${qr.n}" }
        val y = workspace?.take(qr.m) ?: DoubleArray(qr.m)
        applyQInto(qr, b, y, transpose = true)
        y.copyInto(out, 0, 0, qr.n)
        trsvCore(vectorKernels, qr.qr, qr.n, out, lda = qr.m, lower = false, transpose = false, unitDiag = false)
        workspace?.release(y)
        return out
    }

    /** Minimum-norm solution of the consistent underdetermined system `A · x = b` for a wide `m×n` A of full
     *  row rank. Pass the factorization `qr(Aᵀ)`; rank deficiency is not detected. */
    public fun solveMinimumNorm(qr: QrDecomposition, b: DoubleArray, workspace: Workspace? = null): DoubleArray =
        solveMinimumNormInto(qr, b, DoubleArray(qr.m), workspace)

    /** Minimum-norm solve into [out], which has length `m` (the wide system's column count) and is returned.
     *  A [workspace] lends the length-`n` intermediate. */
    public fun solveMinimumNormInto(
        qr: QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray {
        requireShape(qr.m >= qr.n) { "solveMinimumNorm expects the QR of the transpose (tall), got ${qr.m}x${qr.n}" }
        requireShape(b.size == qr.n) { "solveMinimumNorm: b length ${b.size} != ${qr.n}" }
        requireShape(out.size == qr.m) { "solveMinimumNorm: out length ${out.size} != ${qr.m}" }
        val w = workspace?.take(qr.n) ?: DoubleArray(qr.n)
        b.copyInto(w)
        trsvCore(vectorKernels, qr.qr, qr.n, w, lda = qr.m, lower = false, transpose = true, unitDiag = false)
        out.fill(0.0)
        w.copyInto(out)
        workspace?.release(w)
        return applyQInto(qr, out, out)
    }

    /** Order-of-magnitude estimate of `1 / (anorm · est(‖A⁻¹‖₁))` (LAPACK `dgecon`), where [anorm] is the
     *  1-norm of the unfactored matrix (see [norm1]). Returns `1.0` when `n == 0` and `0.0` when singular. */
    public fun rcond(lu: LuDecomposition, anorm: Double, workspace: Workspace? = null): Double {
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        val x = workspace?.take(n) ?: DoubleArray(n)
        val y = workspace?.take(n) ?: DoubleArray(n)
        val signs = workspace?.take(n) ?: DoubleArray(n)
        val probe = workspace?.take(n) ?: DoubleArray(n)
        try {
            return hagerEstimate(lu, anorm, n, x, y, signs, probe, workspace)
        } finally {
            workspace?.release(x)
            workspace?.release(y)
            workspace?.release(signs)
            workspace?.release(probe)
        }
    }

    /** The estimator body, over caller-supplied vectors so [rcond] can pool them. */
    @Suppress("LongParameterList") // four scratch vectors the caller owns
    private fun hagerEstimate(
        lu: LuDecomposition,
        anorm: Double,
        n: Int,
        x: DoubleArray,
        y: DoubleArray,
        signs: DoubleArray,
        probe: DoubleArray,
        workspace: Workspace?,
    ): Double {
        for (i in 0 until n) x[i] = 1.0 / n
        var estimate = 0.0
        var lastPivot = -1
        var sweep = 0
        while (sweep < RCOND_MAX_SWEEPS) {
            solveInto(lu, x, y, workspace = workspace)
            var e = 0.0
            for (i in 0 until n) e += abs(y[i])
            if (e > estimate) estimate = e
            for (i in 0 until n) signs[i] = if (y[i] >= 0.0) 1.0 else -1.0
            val z = solveInto(lu, signs, y, transpose = true, workspace = workspace)
            var j = 0
            for (i in 1 until n) if (abs(z[i]) > abs(z[j])) j = i
            var zx = 0.0
            for (i in 0 until n) zx += z[i] * x[i]
            if (j == lastPivot || abs(z[j]) <= zx) break
            x.fill(0.0)
            x[j] = 1.0
            lastPivot = j
            sweep++
        }
        // The alternating-sign safeguard from LAPACK's dlacn2.
        for (i in 0 until n) {
            probe[i] = (if (i % 2 == 0) 1.0 else -1.0) * (1.0 + i.toDouble() / maxOf(1, n - 1))
        }
        val alt = solveInto(lu, probe, y, workspace = workspace)
        var e = 0.0
        for (i in 0 until n) e += abs(alt[i])
        e = 2.0 * e / (3.0 * n)
        if (e > estimate) estimate = e
        val denominator = estimate * anorm
        return if (denominator.isFinite() && denominator > 0.0) 1.0 / denominator else 0.0
    }

    /** Cholesky factorization `A = L·Lᵀ` of a symmetric positive-definite [a] (`dpotrf` with `uplo = 'L'`).
     *  Reads only the lower triangle of [a], so an upper-only matrix factors to silent nonsense.
     *  @throws com.eignex.koblas.NotPositiveDefinite at the first non-positive pivot unless [policy] allows it.
     */
    public fun cholesky(a: DenseMatrix, policy: CholeskyPolicy = CholeskyPolicy.Strict): CholeskyDecomposition {
        requireShape(a.rows == a.cols) { "cholesky requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        val l = DenseMatrix(n, n)
        val ld = l.data
        val kernels = vectorKernels
        for (j in 0 until n) a.data.copyInto(ld, j + j * n, j + j * n, (j + 1) * n)
        // Right-looking column Cholesky.
        for (j in 0 until n) {
            val base = j + j * n
            val len = n - j
            for (p in 0 until j) {
                val f = ld[j + p * n]
                if (f != 0.0) kernels.axpy(ld, base, -f, ld, j + p * n, len)
            }
            val pivot = ld[base]
            if (pivot <= 0.0 || pivot.isNaN()) {
                if (policy !is CholeskyPolicy.Regularize) {
                    throw NotPositiveDefinite(
                        j,
                        pivot,
                        "matrix is not positive-definite at pivot $j (diagonal=$pivot); pass " +
                            "CholeskyPolicy.Regularize to factor a nearby matrix instead",
                    )
                }
                ld[base] = sqrt(policy.minimumPivot)
            } else {
                ld[base] = sqrt(pivot)
            }
            val diag = ld[base]
            for (i in base + 1 until base + len) ld[i] = ld[i] / diag
        }
        return CholeskyDecomposition(l)
    }

    /** Solve `A · x = b` for the Cholesky factorization [chol] (LAPACK `dpotrs`). [b] is not modified. */
    public fun solve(chol: CholeskyDecomposition, b: DoubleArray): DoubleArray {
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
     *  @throws com.eignex.koblas.SingularMatrix if [lu] is singular; the position is [LuDecomposition.failedAt].
     */
    public fun invert(lu: LuDecomposition, workspace: Workspace? = null): DenseMatrix {
        if (lu.singular) {
            throw SingularMatrix(
                lu.failedAt,
                "invert: factorization is singular at pivot ${lu.failedAt}, so the inverse does not exist",
            )
        }
        val n = lu.n
        val inv = DenseMatrix.diagonal(n)
        return solveInto(lu, inv, inv, transpose = false, workspace = workspace)
    }

    /** Invert a triangular matrix into a fresh result (LAPACK `dtrtri`), returning `T⁻¹` for the [lower] or
     *  upper triangle of the square [a], taking the diagonal as 1 when [unitDiag].
     *  @throws com.eignex.koblas.SingularMatrix naming the first zero diagonal position.
     */
    public fun trtri(a: DenseMatrix, lower: Boolean, unitDiag: Boolean = false): DenseMatrix {
        requireShape(a.rows == a.cols) { "trtri requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        if (!unitDiag) {
            for (i in 0 until n) {
                if (a[i, i] == 0.0) {
                    throw SingularMatrix(i, "trtri: triangle is singular, diagonal entry $i is zero")
                }
            }
        }
        val inv = DenseMatrix(n, n)
        val invd = inv.data
        val column = DoubleArray(n)
        for (j in 0 until n) {
            column.fill(0.0)
            column[j] = 1.0
            trsvCore(
                vectorKernels,
                a.data,
                n,
                column,
                lower = lower,
                transpose = false,
                unitDiag = unitDiag,
            )
            column.copyInto(invd, n * j)
        }
        return inv
    }

    /** Invert an SPD matrix from its Cholesky factorization, returning `A⁻¹` given [chol] (LAPACK `dpotri`). */
    public fun invert(chol: CholeskyDecomposition, workspace: Workspace? = null): DenseMatrix {
        val n = chol.n
        val ld = chol.l.data
        val inv = DenseMatrix(n, n)
        val invd = inv.data
        val kernels = vectorKernels
        val y = workspace?.take(n) ?: DoubleArray(n)
        for (j in 0 until n) {
            y.fill(0.0, j, n)
            y[j] = 1.0
            for (c in j until n) {
                val base = c + c * n
                val yc = y[c] / ld[base]
                y[c] = yc
                if (yc != 0.0) kernels.axpy(y, c + 1, -yc, ld, base + 1, n - c - 1)
            }
            for (i in n - 1 downTo j) {
                val base = i + i * n
                y[i] = (y[i] - kernels.dot(ld, base + 1, y, i + 1, n - i - 1)) / ld[base]
            }
            for (i in j until n) {
                invd[i + j * n] = y[i]
                invd[j + i * n] = y[i]
            }
        }
        workspace?.release(y)
        return inv
    }

    /** [solve] over a [DenseVector] right-hand side. */
    public fun solve(lu: LuDecomposition, b: DenseVector, transpose: Boolean = false): DenseVector =
        DenseVector.wrap(solve(lu, b.data, transpose))

    /** [solveInto] over [DenseVector] operands, returning [out]. */
    public fun solveInto(
        lu: LuDecomposition,
        b: DenseVector,
        out: DenseVector,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DenseVector {
        solveInto(lu, b.data, out.data, transpose, workspace)
        return out
    }

    /** [solve] over a [DenseVector] right-hand side. */
    public fun solve(ldl: LdlDecomposition, b: DenseVector): DenseVector = DenseVector.wrap(solve(ldl, b.data))

    /** [solveInto] over [DenseVector] operands, returning [out]. */
    public fun solveInto(ldl: LdlDecomposition, b: DenseVector, out: DenseVector): DenseVector {
        solveInto(ldl, b.data, out.data)
        return out
    }

    /** [solve] over a [DenseVector] right-hand side. */
    public fun solve(chol: CholeskyDecomposition, b: DenseVector): DenseVector = DenseVector.wrap(solve(chol, b.data))

    /** [solveLeastSquares] over a [DenseVector] right-hand side. */
    public fun solveLeastSquares(qr: QrDecomposition, b: DenseVector, workspace: Workspace? = null): DenseVector =
        DenseVector.wrap(solveLeastSquares(qr, b.data, workspace))

    /** [solveLeastSquares] over a [DenseVector] right-hand side. */
    public fun solveLeastSquares(
        qr: PivotedQrDecomposition,
        b: DenseVector,
        workspace: Workspace? = null,
    ): DenseVector = DenseVector.wrap(solveLeastSquares(qr, b.data, workspace))

    /** [solveMinimumNorm] over a [DenseVector] right-hand side. */
    public fun solveMinimumNorm(qr: QrDecomposition, b: DenseVector, workspace: Workspace? = null): DenseVector =
        DenseVector.wrap(solveMinimumNorm(qr, b.data, workspace))

    /** [applyQ] over a [DenseVector]. */
    public fun applyQ(qr: QrDecomposition, y: DenseVector, transpose: Boolean = false): DenseVector =
        DenseVector.wrap(applyQ(qr, y.data, transpose))
}

/** Sweep cap for the [Lapack.rcond] estimator. */
private const val RCOND_MAX_SWEEPS = 5

/** Run [solveOne] over each column of [b], reassembling the results into an `n × nrhs` matrix. */
@Suppress("LongParameterList") // shape, destination and scratch alongside the per-column solve
private inline fun solveColumnwise(
    b: DenseMatrix,
    out: DenseMatrix,
    n: Int,
    nrhs: Int,
    workspace: Workspace?,
    solveOne: (column: DoubleArray, destination: DoubleArray) -> DoubleArray,
): DenseMatrix {
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

/** dst(i, c) = src(piv(i), c) over an `n × nrhs` column-major pair. [dst] must not alias [src]. */
private fun gatherRows(src: DoubleArray, dst: DoubleArray, n: Int, nrhs: Int, piv: IntArray) {
    for (c in 0 until nrhs) {
        val base = c * n
        for (i in 0 until n) dst[base + i] = src[base + piv[i]]
    }
}

/** dst(piv(i), c) = src(i, c), the inverse of [gatherRows]. */
private fun scatterRows(src: DoubleArray, dst: DoubleArray, n: Int, nrhs: Int, piv: IntArray) {
    for (c in 0 until nrhs) {
        val base = c * n
        for (i in 0 until n) dst[base + piv[i]] = src[base + i]
    }
}
