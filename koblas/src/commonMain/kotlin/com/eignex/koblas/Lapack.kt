@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas

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
        // A narrow block updates rows in `nrhs`-length runs, too short for the vectorized primitives to
        // pay off — at `nrhs` 1 that is one length-1 axpy per matrix element, measured 10x slower than
        // solving column by column. Wide blocks amortize the factor traversal and win from ~32 columns.
        if (nrhs < BLOCK_SOLVE_MIN_RHS) {
            return solveColumnwise(b, out, n, nrhs, workspace) { col, dst ->
                solveInto(lu, col, dst, transpose, workspace)
            }
        }
        val f = lu.lu
        return if (transpose) {
            // Aᵀ X = B, with Aᵀ = Uᵀ Lᵀ P: forward-solve Uᵀ, back-solve unit Lᵀ, un-permute rows.
            val y = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
            b.data.copyInto(y)
            trsmCore(f, n, y, nrhs, lower = false, transpose = true, unitDiag = false)
            trsmCore(f, n, y, nrhs, lower = true, transpose = true, unitDiag = true)
            for (i in 0 until n) y.copyInto(out.data, lu.piv[i] * nrhs, i * nrhs, (i + 1) * nrhs)
            workspace?.release(y)
            out
        } else {
            // A X = B, with P A = L U: permute the rows of B, forward-solve unit L, back-solve U. The
            // gather cannot read B in place once out aliases it, so that case stages.
            if (out === b) {
                val staged = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
                for (i in 0 until n) b.data.copyInto(staged, i * nrhs, lu.piv[i] * nrhs, (lu.piv[i] + 1) * nrhs)
                staged.copyInto(out.data)
                workspace?.release(staged)
            } else {
                for (i in 0 until n) b.data.copyInto(out.data, i * nrhs, lu.piv[i] * nrhs, (lu.piv[i] + 1) * nrhs)
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
     * indefinite factorization (LAPACK `dsytrs` with `nrhs`); returns a fresh `X`. The default is the
     * two-pass `dsytrs` replay over the shared packed format with every scalar step widened to a row;
     * backends may substitute a native block solve.
     */
    @Suppress("CyclomaticComplexMethod", "LongMethod") // dsytrs's control flow, kept recognizable
    fun solve(ldl: LdlDecomposition, b: DenseMatrix): DenseMatrix = solveInto(ldl, b, DenseMatrix(b.rows, b.cols))

    /** Solve `A · X = B` into [out], which is returned. [out] may be [b]. */
    fun solveInto(ldl: LdlDecomposition, b: DenseMatrix, out: DenseMatrix, workspace: Workspace? = null): DenseMatrix {
        val n = ldl.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        // Narrow blocks solve column by column, as in the LU case above.
        require(out.rows == n && out.cols == nrhs) {
            "solve: out is ${out.rows}x${out.cols}, expected ${n}x$nrhs"
        }
        if (nrhs in 1 until BLOCK_SOLVE_MIN_RHS) {
            return solveColumnwise(b, out, n, nrhs, workspace) { col, dst -> solveInto(ldl, col, dst) }
        }
        val w = ldl.ldl
        val ipiv = ldl.ipiv
        val x = out.data
        if (out !== b) b.data.copyInto(x)
        if (nrhs == 0) return out
        // Forward: replay interchanges, apply L block columns, divide by the D blocks (dsytrs lower).
        var k = 0
        while (k < n) {
            if (ipiv[k] > 0) {
                val kp = ipiv[k] - 1
                if (kp != k) swapSolveRows(x, nrhs, k, kp)
                for (i in k + 1 until n) {
                    val f = w[i * n + k]
                    if (f != 0.0) denseAxpy(x, i * nrhs, -f, x, k * nrhs, nrhs)
                }
                denseScale(x, k * nrhs, 1.0 / w[k * n + k], nrhs)
                k += 1
            } else {
                val kp = -ipiv[k] - 1
                if (kp != k + 1) swapSolveRows(x, nrhs, k + 1, kp)
                for (i in k + 2 until n) {
                    val f = w[i * n + k]
                    val f1 = w[i * n + k + 1]
                    if (f != 0.0) denseAxpy(x, i * nrhs, -f, x, k * nrhs, nrhs)
                    if (f1 != 0.0) denseAxpy(x, i * nrhs, -f1, x, (k + 1) * nrhs, nrhs)
                }
                val akm1k = w[(k + 1) * n + k]
                val akm1 = w[k * n + k] / akm1k
                val ak = w[(k + 1) * n + k + 1] / akm1k
                val denom = akm1 * ak - 1.0
                for (c in 0 until nrhs) {
                    val bkm1 = x[k * nrhs + c] / akm1k
                    val bk = x[(k + 1) * nrhs + c] / akm1k
                    x[k * nrhs + c] = (ak * bkm1 - bk) / denom
                    x[(k + 1) * nrhs + c] = (akm1 * bk - bkm1) / denom
                }
                k += 2
            }
        }
        // Backward: apply Lᵀ block rows and undo the interchanges in reverse.
        k = n - 1
        while (k >= 0) {
            if (ipiv[k] > 0) {
                for (i in k + 1 until n) {
                    val f = w[i * n + k]
                    if (f != 0.0) denseAxpy(x, k * nrhs, -f, x, i * nrhs, nrhs)
                }
                val kp = ipiv[k] - 1
                if (kp != k) swapSolveRows(x, nrhs, k, kp)
                k -= 1
            } else {
                val k0 = k - 1
                for (i in k + 1 until n) {
                    val f0 = w[i * n + k0]
                    val f1 = w[i * n + k]
                    if (f0 != 0.0) denseAxpy(x, k0 * nrhs, -f0, x, i * nrhs, nrhs)
                    if (f1 != 0.0) denseAxpy(x, k * nrhs, -f1, x, i * nrhs, nrhs)
                }
                val kp = -ipiv[k] - 1
                if (kp != k) swapSolveRows(x, nrhs, k, kp)
                k -= 2
            }
        }
        return out
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
        // The top n rows of the packed buffer are exactly an n×n row-major block holding R.
        trsvCore(qr.qr, qr.n, out, lower = false, transpose = false, unitDiag = false)
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
        // The top n rows of the packed buffer are exactly an n×n row-major block holding R.
        trsvCore(qr.qr, qr.n, w, lower = false, transpose = true, unitDiag = false)
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
        // Seed l's lower triangle with A's entries (bulk row copies for a dense source), then eliminate
        // in place: each seeded A[i,j] is consumed exactly when that slot is overwritten with l[i,j].
        // Keeps the hot loop free of per-element MatrixView dispatch.
        if (a is DenseMatrix) {
            for (i in 0 until n) a.data.copyInto(ld, i * n, i * n, i * n + i + 1)
        } else {
            for (i in 0 until n) for (j in 0..i) ld[i * n + j] = a[i, j]
        }
        for (i in 0 until n) {
            val rowI = i * n
            for (j in 0..i) {
                val rowJ = j * n
                val sum = denseDot(ld, rowI, ld, rowJ, j)
                if (i == j) {
                    ld[rowI + i] = sqrt(ld[rowI + i] - sum)
                } else {
                    ld[rowI + j] = (ld[rowI + j] - sum) / ld[rowJ + j]
                }
            }
            if (ld[rowI + i] <= 0.0 || ld[rowI + i].isNaN()) {
                require(regularizeNonPD) {
                    "matrix is not positive-definite at pivot $i (diagonal=${ld[rowI + i]})"
                }
                ld[rowI + i] = 1e-5
            }
        }
        return l
    }

    /**
     * Solve `A * x = b` for `x`, given `L = chol(A)` (lower-triangular, `A = L * LT`).
     * Allocates a fresh result vector; [b] is not modified.
     *
     * Forward substitution `L * y = b` runs over contiguous row data and uses [denseDot].
     * Back substitution `LT * x = y` is column-oriented: once `x[i]` is final, its contribution
     * is subtracted from the remaining right-hand side along contiguous row `i` via [denseAxpy].
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
            // L y = e_j: rows before j stay zero.
            y[j] = 1.0 / ld[j * n + j]
            for (i in j + 1 until n) {
                val rowI = i * n
                y[i] = -denseDot(ld, rowI + j, y, j, i - j) / ld[rowI + i]
            }
            // LT x = y, column-oriented, restricted to rows >= j.
            for (i in n - 1 downTo j) {
                val xi = y[i] / ld[i * n + i]
                y[i] = xi
                if (xi != 0.0) denseAxpy(y, j, -xi, ld, i * n + j, i - j)
            }
            for (i in j until n) {
                invd[i * n + j] = y[i]
                invd[j * n + i] = y[i]
            }
        }
        workspace?.release(y)
        return inv
    }
}

/** Sweep cap for the [Lapack.rcond] estimator; Hager's iteration almost always converges in 2. */
private const val RCOND_MAX_SWEEPS = 5

/**
 * Right-hand-side count from which a block solve beats solving column by column. Below it the block
 * traversal updates rows in runs too short to vectorize; above it the single pass over the factor pays
 * for itself. Measured crossover at `n` 64 and 256: column solves win through 16 columns, the block
 * wins from 32.
 */
private const val BLOCK_SOLVE_MIN_RHS = 32

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
    // One column in, one column out: two n-vectors serve every right-hand side.
    val col = workspace?.take(n) ?: DoubleArray(n)
    val solved = workspace?.take(n) ?: DoubleArray(n)
    for (c in 0 until nrhs) {
        for (i in 0 until n) col[i] = b.data[i * nrhs + c]
        val x = solveOne(col, solved)
        for (i in 0 until n) out.data[i * nrhs + c] = x[i]
    }
    if (workspace != null) {
        workspace.release(solved)
        workspace.release(col)
    }
    return out
}

/** Swap rows [r1] and [r2] of a flat row-major solve buffer with [nrhs] columns. */
private fun swapSolveRows(x: DoubleArray, nrhs: Int, r1: Int, r2: Int) {
    for (c in 0 until nrhs) {
        val t = x[r1 * nrhs + c]
        x[r1 * nrhs + c] = x[r2 * nrhs + c]
        x[r2 * nrhs + c] = t
    }
}
