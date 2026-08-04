@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.MatrixView
import com.eignex.koblas.Workspace
import com.eignex.koblas.norm1
import com.eignex.koblas.transpose
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The factorizations and the solves built on them, the seam a native LAPACK plugs into.
 *
 * These carry English names rather than LAPACK's mnemonics: `factor` and `cholesky` read better than
 * `getrf` and `potrf`, and unlike the BLAS mnemonics those abbreviations are opaque even to people who
 * know the libraries. That applies to the most-used entry points too — [factor] and [solve] stay English
 * rather than becoming `getrf` and `getrs` — so the rule holds without exception on this side of the
 * seam. The packed formats are LAPACK's, so a factorization produced by one backend solves correctly on
 * another.
 *
 * Ranked and selected separately from [Blas], because a host can provide one and not the other.
 * Defaults implement every routine in portable Kotlin, so a backend overrides only what it accelerates.
 */
interface Lapack : Backend {
    /** LU factorization with partial pivoting of a square [a] (LAPACK `dgetrf`); [a] is not modified.
     *  Allocates the factor buffers; [factorInto] refactorizes into existing ones. */
    fun factor(a: DenseMatrix): LuDecomposition

    /**
     * Refactorize [a] into [out]'s existing buffers, returning [out]. A periodic refactorization — the
     * simplex rebuilding its basis, a filter re-decomposing a covariance — otherwise allocates an `n²`
     * copy and a pivot array each time; this reuses both. [out] must have the same dimension as [a], and
     * its previous contents are discarded.
     */
    fun factorInto(a: DenseMatrix, out: LuDecomposition): LuDecomposition {
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        require(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        // The default has no way to reuse a backend's internals, so it factors and copies the result in.
        val fresh = factor(a)
        fresh.lu.copyInto(out.lu)
        fresh.piv.copyInto(out.piv)
        out.singular = fresh.singular
        return out
    }

    /** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for the factorization [lu] (LAPACK `dgetrs`);
     *  returns a fresh `x`. [transpose] serves the simplex's BTRAN (`Bᵀ y = c`) against a factored basis.
     *  Allocates the result; [solveInto] writes into a caller-owned destination instead. */
    fun solve(lu: LuDecomposition, b: DoubleArray, transpose: Boolean = false): DoubleArray =
        solveInto(lu, b, DoubleArray(lu.n), transpose)

    /**
     * Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) into [out], which is returned. Nothing is
     * allocated, so a loop that owns its destination — a simplex FTRAN/BTRAN against a factored basis, a
     * filter update — runs without touching the collector. [out] may be the same array as [b].
     *
     * The transposed direction has to stage the solved vector before scattering it through the
     * permutation, so pass a [workspace] to make that staging buffer reusable as well; without one it is
     * the single allocation this routine still makes.
     */
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    fun solveInto(
        lu: LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray

    /**
     * Solve `A · X = B` (or `Aᵀ · X = B` when [transpose]) for the [b].cols right-hand-side columns of
     * [b] at once (LAPACK `dgetrs` with `nrhs`); returns a fresh `X`. The default runs the permutation
     * and the two triangular block solves directly on the shared packed format; backends may
     * substitute a native block solve.
     */
    fun solve(lu: LuDecomposition, b: DenseMatrix, transpose: Boolean = false): DenseMatrix =
        solveInto(lu, b, DenseMatrix(b.rows, b.cols), transpose)

    /**
     * Solve `A · X = B` (or `Aᵀ · X = B` when [transpose]) into [out], which is returned. [out] may be
     * [b]. The transposed direction stages a block before scattering its rows through the permutation,
     * so pass a [workspace] to lend that `n·nrhs` buffer.
     */
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    fun solveInto(
        lu: LuDecomposition,
        b: DenseMatrix,
        out: DenseMatrix,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DenseMatrix {
        val n = lu.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        require(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        if (nrhs == 0) return out
        // One code path for every `nrhs`. Under row-major storage a narrow block updated rows in
        // `nrhs`-length runs, too short to vectorize, so anything below 32 columns had to be solved
        // column by column instead; here the right-hand sides *are* the contiguous columns, so the block
        // form is already the column form and the split has nothing left to choose between.
        val f = lu.lu
        return if (transpose) {
            // Aᵀ X = B, with Aᵀ = Uᵀ Lᵀ P: forward-solve Uᵀ, back-solve unit Lᵀ, un-permute rows.
            val y = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
            b.data.copyInto(y)
            trsmCore(f, n, y, nrhs, lower = false, transpose = true, unitDiag = false)
            trsmCore(f, n, y, nrhs, lower = true, transpose = true, unitDiag = true)
            scatterRows(y, out.data, n, nrhs, lu.piv)
            workspace?.release(y)
            out
        } else {
            // A X = B, with P A = L U: permute the rows of B, forward-solve unit L, back-solve U. The
            // gather cannot read B in place once out aliases it, so that case stages.
            if (out === b) {
                val staged = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
                gatherRows(b.data, staged, n, nrhs, lu.piv)
                staged.copyInto(out.data)
                workspace?.release(staged)
            } else {
                gatherRows(b.data, out.data, n, nrhs, lu.piv)
            }
            trsmCore(f, n, out.data, nrhs, lower = true, transpose = false, unitDiag = true)
            trsmCore(f, n, out.data, nrhs, lower = false, transpose = false, unitDiag = false)
            out
        }
    }

    /**
     * Symmetric indefinite factorization `A = L·D·Lᵀ` with Bunch–Kaufman partial pivoting (LAPACK
     * `dsytrf`, lower). As with [Blas.symv], only the lower triangle of [a] is read — the strictly upper
     * triangle may hold anything — and [a] is not modified. Use this where the matrix is symmetric but
     * not positive definite (KKT systems); for SPD matrices [cholesky] is cheaper.
     */
    fun ldl(a: DenseMatrix, workspace: Workspace? = null): LdlDecomposition

    /** Solve `A · x = b` for a symmetric indefinite factorization [ldl] (LAPACK `dsytrs`); returns a
     *  fresh `x`. Symmetry makes the transposed solve identical, so there is no `transpose` flag.
     *  Allocates the result; [solveInto] writes into a caller-owned destination instead. */
    fun solve(ldl: LdlDecomposition, b: DoubleArray): DoubleArray = solveInto(ldl, b, DoubleArray(ldl.n))

    /** Solve `A · x = b` into [out], which is returned; allocates nothing. [out] may be [b]. */
    fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray

    /**
     * Solve `A · X = B` for the [b].cols right-hand-side columns of [b] at once against a symmetric
     * indefinite factorization (LAPACK `dsytrs` with `nrhs`); returns a fresh `X`. Backends may
     * substitute a native block solve.
     */
    fun solve(ldl: LdlDecomposition, b: DenseMatrix): DenseMatrix = solveInto(ldl, b, DenseMatrix(b.rows, b.cols))

    /**
     * Solve `A · X = B` into [out], which is returned. [out] may be [b].
     *
     * One `dsytrs` replay per right-hand side, over the contiguous column it occupies. Under row-major
     * storage this was a single widened pass, because extracting a column meant a strided gather; here a
     * column is already a contiguous run, so reusing the tuned vector solve costs nothing but re-reading
     * the factor and keeps the pivot-block bookkeeping in one place instead of two.
     */
    fun solveInto(ldl: LdlDecomposition, b: DenseMatrix, out: DenseMatrix, workspace: Workspace? = null): DenseMatrix {
        val n = ldl.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        require(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        return solveColumnwise(b, out, n, nrhs, workspace) { col, dst -> solveInto(ldl, col, dst) }
    }

    /**
     * QR factorization `A = Q·R` of an `m×n` [a] via Householder reflections (LAPACK `dgeqrf`); [a] is
     * not modified and any shape is accepted. Rank deficiency is not detected — zero diagonal entries
     * of `R` surface in [solveLeastSquares] as infinities/NaNs, following the triangular-solve
     * convention.
     */
    fun qr(a: DenseMatrix, workspace: Workspace? = null): QrDecomposition

    /** Apply `Q` (or `Qᵀ` when [transpose]) from [qr] to a length-`m` [y], into a fresh result
     *  (LAPACK `dormqr` restricted to a single column). Allocates; [applyQInto] does not. */
    fun applyQ(qr: QrDecomposition, y: DoubleArray, transpose: Boolean = false): DoubleArray =
        applyQInto(qr, y, DoubleArray(qr.m), transpose)

    /** Apply `Q` (or `Qᵀ` when [transpose]) from [qr] to [y] into [out], which is returned. [out] may
     *  be [y]. */
    fun applyQInto(qr: QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean = false): DoubleArray

    /**
     * Least-squares solve `min ‖A·x − b‖₂` from the factorization (the `dgels` shape): requires
     * `m ≥ n` and full column rank, returns the length-`n` solution `x = R⁻¹·(Qᵀb)[0..n)`. This is
     * the kernel square-root/array filters build on; it composes with [cholesky] rank-one updates for
     * sliding-window problems.
     */
    fun solveLeastSquares(qr: QrDecomposition, b: DoubleArray, workspace: Workspace? = null): DoubleArray =
        solveLeastSquaresInto(qr, b, DoubleArray(qr.n), workspace)

    /**
     * Least-squares solve into [out], which has length `n` and is returned. The `Qᵀb` product needs a
     * length-`m` intermediate, so pass a [workspace] to make the call allocation-free.
     */
    fun solveLeastSquaresInto(
        qr: QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray {
        require(qr.m >= qr.n) { "solveLeastSquares requires m >= n, got ${qr.m}x${qr.n}" }
        require(b.size == qr.m) { "solveLeastSquares: b length ${b.size} != ${qr.m}" }
        require(out.size == qr.n) { "solveLeastSquares: out length ${out.size} != ${qr.n}" }
        val y = workspace?.take(qr.m) ?: DoubleArray(qr.m)
        applyQInto(qr, b, y, transpose = true)
        y.copyInto(out, 0, 0, qr.n)
        // R is the leading n×n triangle of the packed m×n buffer, so it is read at the buffer's own
        // leading dimension rather than as a standalone n×n block.
        trsvCore(qr.qr, qr.n, out, lda = qr.m, lower = false, transpose = false, unitDiag = false)
        workspace?.release(y)
        return out
    }

    /**
     * Minimum-norm solution of the underdetermined consistent system `A · x = b` for a wide `m×n` A
     * with `m <= n` and full row rank (LAPACK `dgels`'s underdetermined shape, via QR of the transpose
     * instead of LQ): pass the factorization `qr(Aᵀ)`. With `Aᵀ = Q·R` we have `A = Rᵀ·Qᵀ`, so a
     * forward solve `Rᵀ·w = b` followed by `x = Q·(w padded with zeros)` gives the solution of
     * smallest 2-norm. [b] has length `m`; the result has length `n`. Rank deficiency is not detected
     * and surfaces as infinities/NaNs, following the triangular-solve convention.
     */
    fun solveMinimumNorm(qr: QrDecomposition, b: DoubleArray, workspace: Workspace? = null): DoubleArray =
        solveMinimumNormInto(qr, b, DoubleArray(qr.m), workspace)

    /**
     * Minimum-norm solve into [out], which has length `m` (the wide system's column count) and is
     * returned. A length-`n` intermediate holds the forward solve, so pass a [workspace] to make the
     * call allocation-free.
     */
    fun solveMinimumNormInto(
        qr: QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace? = null,
    ): DoubleArray {
        require(qr.m >= qr.n) { "solveMinimumNorm expects the QR of the transpose (tall), got ${qr.m}x${qr.n}" }
        require(b.size == qr.n) { "solveMinimumNorm: b length ${b.size} != ${qr.n}" }
        require(out.size == qr.m) { "solveMinimumNorm: out length ${out.size} != ${qr.m}" }
        val w = workspace?.take(qr.n) ?: DoubleArray(qr.n)
        b.copyInto(w)
        // R is the leading n×n triangle of the packed m×n buffer; see [solveLeastSquaresInto].
        trsvCore(qr.qr, qr.n, w, lda = qr.m, lower = false, transpose = true, unitDiag = false)
        // Pad to length m with zeros, then apply Q in place.
        out.fill(0.0)
        w.copyInto(out)
        workspace?.release(w)
        return applyQInto(qr, out, out)
    }

    /**
     * Reciprocal condition number estimate `1 / (anorm · est(‖A⁻¹‖₁))` from a factorization (LAPACK
     * `dgecon`). [anorm] is the 1-norm of the original, unfactored matrix (see [norm1]), which the
     * caller computes before factoring. Returns `1.0` for the empty factorization and exactly `0.0`
     * when [lu] is singular or [anorm] is zero.
     *
     * This is an order-of-magnitude estimate, not an exact condition number: the estimator never
     * exceeds the true `‖A⁻¹‖₁`, so the returned value never understates the conditioning. It is the
     * cheap signal a revised simplex uses to decide when to refactorize. The default implementation is
     * a Hager-style 1-norm estimator over [solveInto]; backends may substitute a native estimator, so the
     * exact value can differ between backends while agreeing in magnitude.
     *
     * The estimator needs four vectors and runs several sweeps, so it is the allocation-heaviest routine
     * here — which matters because a simplex calls it to decide when to refactorize. Passing a
     * [workspace] reuses those buffers across calls and makes it allocation-free.
     */
    fun rcond(lu: LuDecomposition, anorm: Double, workspace: Workspace? = null): Double {
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        // Hager's power iteration on the 1-norm of A⁻¹: follow sign vectors through A⁻¹ and A⁻ᵀ,
        // restarting from the unit vector of the strongest coordinate until it stops improving.
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
            // z reuses y: the sign vector is already read out of it.
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
        // The alternating-sign safeguard from LAPACK's dlacn2 catches matrices the power step misses;
        // its probe vector has 1-norm 3n/2, hence the 2/(3n) normalization.
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

    /**
     * Lower-triangular Cholesky decomposition `A = L * LT`, returned as a fresh matrix.
     *
     * With [regularizeNonPD] true (default), non-positive-definite inputs get a small
     * positive diagonal entry instead of a crash - suitable for the online stats that
     * call this on drifting precision matrices. Pass `false` for strict
     * positive-definiteness validation (e.g. user-supplied prior covariance); the
     * function throws [IllegalArgumentException] at the first non-PD pivot.
     */
    fun cholesky(a: MatrixView, regularizeNonPD: Boolean = true): DenseMatrix {
        require(a.rows == a.cols) { "cholesky requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        val l = DenseMatrix(n, n)
        val ld = l.data
        // Seed l's lower triangle with A's entries (bulk column copies for a dense source), then
        // eliminate in place: each seeded A[i,j] is consumed exactly when that slot is overwritten with
        // l[i,j]. Keeps the hot loop free of per-element MatrixView dispatch.
        if (a is DenseMatrix) {
            for (j in 0 until n) a.data.copyInto(ld, j + j * n, j + j * n, (j + 1) * n)
        } else {
            for (j in 0 until n) for (i in j until n) ld[i + j * n] = a[i, j]
        }
        // Right-looking column Cholesky: subtract the already-computed columns from column j, take the
        // pivot, then scale. Every run is the contiguous tail of a column.
        for (j in 0 until n) {
            val base = j + j * n
            val len = n - j
            for (p in 0 until j) {
                val f = ld[j + p * n]
                if (f != 0.0) denseAxpy(ld, base, -f, ld, j + p * n, len)
            }
            val pivot = ld[base]
            if (pivot <= 0.0 || pivot.isNaN()) {
                require(regularizeNonPD) {
                    "matrix is not positive-definite at pivot $j (diagonal=$pivot)"
                }
                ld[base] = 1e-5
            } else {
                ld[base] = sqrt(pivot)
            }
            val diag = ld[base]
            for (i in base + 1 until base + len) ld[i] = ld[i] / diag
        }
        return l
    }

    /**
     * Solve `A * x = b` for `x`, given `L = chol(A)` (lower-triangular, `A = L * LT`).
     * Allocates a fresh result vector; [b] is not modified.
     *
     * Forward substitution `L * y = b` is column-oriented: once `y[j]` is final, its contribution is
     * subtracted from the remaining right-hand side down contiguous column `j` via [denseAxpy]. Back
     * substitution `LT * x = y` reads row `i` of `LT`, which is column `i` of `L`, so it uses [denseDot]
     * on the same contiguous runs.
     */
    fun solveSpd(L: DenseMatrix, b: DoubleArray): DoubleArray {
        val n = L.rows
        require(b.size == n) { "solveSpd: b size ${b.size}, expected $n" }
        val x = b.copyOf()
        trsvCore(L.data, n, x, lower = true, transpose = false, unitDiag = false)
        trsvCore(L.data, n, x, lower = true, transpose = true, unitDiag = false)
        return x
    }

    /**
     * Invert an SPD matrix from its Cholesky factor: returns `A^-1` given `L = chol(A)`.
     *
     * Solves `A * x = e_j` column by column, exploiting the unit-vector right-hand side: forward
     * substitution starts at row `j` (the leading entries are provably zero) and back substitution
     * only produces rows `>= j` — the strictly-upper entries of the symmetric `A^-1` come from
     * mirroring the lower triangle.
     */
    fun invertSpd(L: DenseMatrix, workspace: Workspace? = null): DenseMatrix {
        val n = L.rows
        val ld = L.data
        val inv = DenseMatrix(n, n)
        val invd = inv.data
        val y = workspace?.take(n) ?: DoubleArray(n)
        for (j in 0 until n) {
            // L y = e_j: rows before j stay zero, so the forward sweep starts at column j.
            y.fill(0.0, j, n)
            y[j] = 1.0
            for (c in j until n) {
                val base = c + c * n
                val yc = y[c] / ld[base]
                y[c] = yc
                if (yc != 0.0) denseAxpy(y, c + 1, -yc, ld, base + 1, n - c - 1)
            }
            // Lᵀ x = y, restricted to rows >= j; column i of L is row i of Lᵀ, so this dots behind the
            // frontier.
            for (i in n - 1 downTo j) {
                val base = i + i * n
                y[i] = (y[i] - denseDot(ld, base + 1, y, i + 1, n - i - 1)) / ld[base]
            }
            for (i in j until n) {
                invd[i + j * n] = y[i]
                invd[j + i * n] = y[i]
            }
        }
        workspace?.release(y)
        return inv
    }
}

/** Sweep cap for the [Lapack.rcond] estimator; Hager's iteration almost always converges in 2. */
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
    // One column in, one column out: two n-vectors serve every right-hand side, and staging a column is
    // a bulk copy rather than a strided gather.
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

/** `dst[i, c] = src[piv[i], c]` over an `n × nrhs` column-major pair; [dst] must not alias [src]. */
private fun gatherRows(src: DoubleArray, dst: DoubleArray, n: Int, nrhs: Int, piv: IntArray) {
    for (c in 0 until nrhs) {
        val base = c * n
        for (i in 0 until n) dst[base + i] = src[base + piv[i]]
    }
}

/** `dst[piv[i], c] = src[i, c]`, the inverse of [gatherRows]. */
private fun scatterRows(src: DoubleArray, dst: DoubleArray, n: Int, nrhs: Int, piv: IntArray) {
    for (c in 0 until nrhs) {
        val base = c * n
        for (i in 0 until n) dst[base + piv[i]] = src[base + i]
    }
}
