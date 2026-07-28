@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas

import kotlin.concurrent.Volatile
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The swappable compute backend for the heavier dense operations — level-2/3 BLAS ([gemv], [gemm]) and a
 * general LU [factor]/[solve] — where a native BLAS/LAPACK implementation is worth dispatching to.
 * Everything is over flat, contiguous [DenseMatrix.data] / [DoubleArray] buffers, so a native backend
 * passes them across the FFI boundary without repacking.
 *
 * Level-1 kernels (`dot`/`axpy`/`scale`) are the compile-time-specialized SIMD `Primitives` seam, not part
 * of this interface: they are too fine-grained for a per-call runtime dispatch and are already
 * vectorized. This interface is the runtime seam — [koblas] resolves to [platformLinearAlgebra] when a
 * target provides a native backend, else the portable [ReferenceLinearAlgebra].
 */
interface LinearAlgebra {
    /** A short backend identifier for diagnostics (e.g. `"reference"`). */
    val name: String

    /**
     * Relative preference among simultaneously available backends: automatic selection — JVM
     * classpath discovery and native [registerLinearAlgebra] — picks the highest. The portable
     * reference is 0; native-accelerated backends rank above it (koblas-openblas 100, koblas-cblas
     * 90).
     */
    val priority: Int get() = 0

    /**
     * In-place matrix-vector accumulate `y = alpha · op(A) · x + beta · y` (full BLAS `dgemv`),
     * where `op(A)` is `Aᵀ` when [transpose]. Per BLAS convention, `beta == 0.0` overwrites [y]
     * without reading it (it may be uninitialized), and `alpha == 0.0` reduces to the `beta` scale.
     */
    fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false)

    /** Matrix-vector product `A · x`, or `Aᵀ · x` when [transpose], into a fresh result (restricted
     *  [gemv] with `alpha = 1, beta = 0`). */
    fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /**
     * In-place matrix-matrix accumulate `C = alpha · op(A) · op(B) + beta · C` (full BLAS `dgemm`),
     * where `op` transposes its operand when [transposeA] / [transposeB] is set. Shapes must satisfy
     * `op(A): m×k`, `op(B): k×n`, `C: m×n`. Per BLAS convention, `beta == 0.0` overwrites [c] without
     * reading it, and `alpha == 0.0` reduces to the `beta` scale.
     */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    )

    /** Matrix-matrix product `A · B` into a fresh matrix (restricted [gemm] with `alpha = 1, beta = 0`);
     *  `A.cols` must equal `B.rows`. */
    fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
    }

    /**
     * In-place symmetric rank-k accumulate `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C`
     * when [transpose] (BLAS `dsyrk`). With the default [Uplo.FULL] the full symmetric result is
     * produced (the alpha term is applied to both triangles, and beta scales all of [c]); with
     * [Uplo.LOWER] / [Uplo.UPPER] the standard `dsyrk` semantics apply — only the selected triangle is
     * written and beta-scaled, the opposite strict triangle untouched. [c] must be square with
     * dimension `op(A).rows`. Per BLAS convention, `beta == 0.0` overwrites without reading (within
     * the written region), and `alpha == 0.0` reduces to the `beta` scale.
     */
    @Suppress("LongParameterList") // the BLAS dsyrk signature
    fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, uplo: Uplo = Uplo.FULL)

    /**
     * In-place symmetric matrix-vector accumulate `y = alpha · A · x + beta · y` for a symmetric [a]
     * (BLAS `dsymv`). Only the triangle selected by [lower] (diagonal included) is read; the opposite
     * strict triangle may hold anything. Exploits symmetry for roughly half the memory traffic of
     * [gemv]. Per BLAS convention, `beta == 0.0` overwrites [y] without reading it, and `alpha == 0.0`
     * reduces to the `beta` scale.
     */
    @Suppress("LongParameterList") // the BLAS dsymv signature
    fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean = true)

    /**
     * In-place symmetric matrix-matrix accumulate `C = alpha · A · B + beta · C`, or
     * `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). As with [symv], only the triangle of
     * the symmetric [a] selected by [lower] is read. Shapes: [b] and [c] agree, and [a] is square with
     * dimension `B.rows` (left) or `B.cols` (right). Per BLAS convention, `beta == 0.0` overwrites [c]
     * without reading it, and `alpha == 0.0` reduces to the `beta` scale.
     */
    @Suppress("LongParameterList") // the BLAS dsymm signature
    fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
    )

    /** LU factorization with partial pivoting of a square [a] (LAPACK `dgetrf`); [a] is not modified. */
    fun factor(a: DenseMatrix): LuDecomposition

    /** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for the factorization [lu] (LAPACK `dgetrs`);
     *  returns a fresh `x`. [transpose] serves the simplex's BTRAN (`Bᵀ y = c`) against a factored basis. */
    fun solve(lu: LuDecomposition, b: DoubleArray, transpose: Boolean = false): DoubleArray

    /**
     * Solve `A · X = B` (or `Aᵀ · X = B` when [transpose]) for the [b].cols right-hand-side columns of
     * [b] at once (LAPACK `dgetrs` with `nrhs`); returns a fresh `X`. The default runs the permutation
     * and the two triangular block solves directly on the shared packed format; backends may
     * substitute a native block solve.
     */
    fun solve(lu: LuDecomposition, b: DenseMatrix, transpose: Boolean = false): DenseMatrix {
        val n = lu.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        val f = lu.lu
        return if (transpose) {
            // Aᵀ X = B, with Aᵀ = Uᵀ Lᵀ P: forward-solve Uᵀ, back-solve unit Lᵀ, un-permute rows.
            val y = b.data.copyOf()
            trsmCore(f, n, y, nrhs, lower = false, transpose = true, unitDiag = false)
            trsmCore(f, n, y, nrhs, lower = true, transpose = true, unitDiag = true)
            val x = DoubleArray(n * nrhs)
            for (i in 0 until n) y.copyInto(x, lu.piv[i] * nrhs, i * nrhs, (i + 1) * nrhs)
            DenseMatrix.wrap(n, nrhs, x)
        } else {
            // A X = B, with P A = L U: permute the rows of B, forward-solve unit L, back-solve U.
            val x = DoubleArray(n * nrhs)
            for (i in 0 until n) b.data.copyInto(x, i * nrhs, lu.piv[i] * nrhs, (lu.piv[i] + 1) * nrhs)
            trsmCore(f, n, x, nrhs, lower = true, transpose = false, unitDiag = true)
            trsmCore(f, n, x, nrhs, lower = false, transpose = false, unitDiag = false)
            DenseMatrix.wrap(n, nrhs, x)
        }
    }

    /**
     * Symmetric indefinite factorization `A = L·D·Lᵀ` with Bunch–Kaufman partial pivoting (LAPACK
     * `dsytrf`, lower). As with [symv], only the lower triangle of [a] is read — the strictly upper
     * triangle may hold anything — and [a] is not modified. Use this where the matrix is symmetric but
     * not positive definite (KKT systems); for SPD matrices [cholesky] is cheaper.
     */
    fun ldl(a: DenseMatrix): LdlDecomposition

    /** Solve `A · x = b` for a symmetric indefinite factorization [ldl] (LAPACK `dsytrs`); returns a
     *  fresh `x`. Symmetry makes the transposed solve identical, so there is no `transpose` flag. */
    fun solve(ldl: LdlDecomposition, b: DoubleArray): DoubleArray

    /**
     * Solve `A · X = B` for the [b].cols right-hand-side columns of [b] at once against a symmetric
     * indefinite factorization (LAPACK `dsytrs` with `nrhs`); returns a fresh `X`. The default is the
     * two-pass `dsytrs` replay over the shared packed format with every scalar step widened to a row;
     * backends may substitute a native block solve.
     */
    @Suppress("CyclomaticComplexMethod", "LongMethod") // dsytrs's control flow, kept recognizable
    fun solve(ldl: LdlDecomposition, b: DenseMatrix): DenseMatrix {
        val n = ldl.n
        val nrhs = b.cols
        require(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        val w = ldl.ldl
        val ipiv = ldl.ipiv
        val x = b.data.copyOf()
        if (nrhs == 0) return DenseMatrix.wrap(n, nrhs, x)
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
        return DenseMatrix.wrap(n, nrhs, x)
    }

    /**
     * QR factorization `A = Q·R` of an `m×n` [a] via Householder reflections (LAPACK `dgeqrf`); [a] is
     * not modified and any shape is accepted. Rank deficiency is not detected — zero diagonal entries
     * of `R` surface in [solveLeastSquares] as infinities/NaNs, following the triangular-solve
     * convention.
     */
    fun qr(a: DenseMatrix): QrDecomposition

    /** Apply `Q` (or `Qᵀ` when [transpose]) from [qr] to a length-`m` [y], into a fresh result
     *  (LAPACK `dormqr` restricted to a single column). */
    fun applyQ(qr: QrDecomposition, y: DoubleArray, transpose: Boolean = false): DoubleArray

    /**
     * Least-squares solve `min ‖A·x − b‖₂` from the factorization (the `dgels` shape): requires
     * `m ≥ n` and full column rank, returns the length-`n` solution `x = R⁻¹·(Qᵀb)[0..n)`. This is
     * the kernel square-root/array filters build on; it composes with [cholesky] rank-one updates for
     * sliding-window problems.
     */
    fun solveLeastSquares(qr: QrDecomposition, b: DoubleArray): DoubleArray {
        require(qr.m >= qr.n) { "solveLeastSquares requires m >= n, got ${qr.m}x${qr.n}" }
        require(b.size == qr.m) { "solveLeastSquares: b length ${b.size} != ${qr.m}" }
        val y = applyQ(qr, b, transpose = true)
        val x = y.copyOf(qr.n)
        // The top n rows of the packed buffer are exactly an n×n row-major block holding R.
        trsvCore(qr.qr, qr.n, x, lower = false, transpose = false, unitDiag = false)
        return x
    }

    /**
     * Minimum-norm solution of the underdetermined consistent system `A · x = b` for a wide `m×n` A
     * with `m <= n` and full row rank (LAPACK `dgels`'s underdetermined shape, via QR of the transpose
     * instead of LQ): pass the factorization `qr(Aᵀ)`. With `Aᵀ = Q·R` we have `A = Rᵀ·Qᵀ`, so a
     * forward solve `Rᵀ·w = b` followed by `x = Q·(w padded with zeros)` gives the solution of
     * smallest 2-norm. [b] has length `m`; the result has length `n`. Rank deficiency is not detected
     * and surfaces as infinities/NaNs, following the triangular-solve convention.
     */
    fun solveMinimumNorm(qr: QrDecomposition, b: DoubleArray): DoubleArray {
        require(qr.m >= qr.n) { "solveMinimumNorm expects the QR of the transpose (tall), got ${qr.m}x${qr.n}" }
        require(b.size == qr.n) { "solveMinimumNorm: b length ${b.size} != ${qr.n}" }
        val w = b.copyOf()
        // The top n rows of the packed buffer are exactly an n×n row-major block holding R.
        trsvCore(qr.qr, qr.n, w, lower = false, transpose = true, unitDiag = false)
        val y = DoubleArray(qr.m)
        w.copyInto(y)
        return applyQ(qr, y)
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
     * a Hager-style 1-norm estimator over [solve]; backends may substitute a native estimator, so the
     * exact value can differ between backends while agreeing in magnitude.
     */
    fun rcond(lu: LuDecomposition, anorm: Double): Double {
        val n = lu.n
        if (n == 0) return 1.0
        if (lu.singular || anorm == 0.0) return 0.0
        // Hager's power iteration on the 1-norm of A⁻¹: follow sign vectors through A⁻¹ and A⁻ᵀ,
        // restarting from the unit vector of the strongest coordinate until it stops improving.
        var x = DoubleArray(n) { 1.0 / n }
        var estimate = 0.0
        var lastPivot = -1
        var sweep = 0
        while (sweep < RCOND_MAX_SWEEPS) {
            val y = solve(lu, x)
            var e = 0.0
            for (v in y) e += abs(v)
            if (e > estimate) estimate = e
            val signs = DoubleArray(n) { i -> if (y[i] >= 0.0) 1.0 else -1.0 }
            val z = solve(lu, signs, transpose = true)
            var j = 0
            for (i in 1 until n) if (abs(z[i]) > abs(z[j])) j = i
            var zx = 0.0
            for (i in 0 until n) zx += z[i] * x[i]
            if (j == lastPivot || abs(z[j]) <= zx) break
            x = DoubleArray(n)
            x[j] = 1.0
            lastPivot = j
            sweep++
        }
        // The alternating-sign safeguard from LAPACK's dlacn2 catches matrices the power step misses;
        // its probe vector has 1-norm 3n/2, hence the 2/(3n) normalization.
        val probe = DoubleArray(n) { i ->
            (if (i % 2 == 0) 1.0 else -1.0) * (1.0 + i.toDouble() / maxOf(1, n - 1))
        }
        val alt = solve(lu, probe)
        var e = 0.0
        for (v in alt) e += abs(v)
        e = 2.0 * e / (3.0 * n)
        if (e > estimate) estimate = e
        val denominator = estimate * anorm
        return if (denominator.isFinite() && denominator > 0.0) 1.0 / denominator else 0.0
    }
}

/** Sweep cap for the [LinearAlgebra.rcond] estimator; Hager's iteration almost always converges in 2. */
private const val RCOND_MAX_SWEEPS = 5

/** Swap rows [r1] and [r2] of a flat row-major solve buffer with [nrhs] columns. */
private fun swapSolveRows(x: DoubleArray, nrhs: Int, r1: Int, r2: Int) {
    for (c in 0 until nrhs) {
        val t = x[r1 * nrhs + c]
        x[r1 * nrhs + c] = x[r2 * nrhs + c]
        x[r2 * nrhs + c] = t
    }
}

/** Bunch–Kaufman pivot threshold `(1 + √17) / 8`, the value minimizing element growth (netlib dsytf2). */
private const val BUNCH_KAUFMAN_ALPHA = 0.6403882032022076

/**
 * A general LU factorization with partial pivoting: `P·A = L·U`, the unit-lower `L` and upper `U` packed
 * into one flat row-major [lu] buffer (`L` below the diagonal, `U` on and above) and the row permutation
 * in [piv] (`piv[k]` is the original row now at position `k`). [singular] is set when a zero pivot was
 * hit; [LinearAlgebra.solve] on a singular factorization is not meaningful. Produced by
 * [LinearAlgebra.factor].
 *
 * The constructor and the factor buffers are public so that out-of-repo [LinearAlgebra] backends can
 * produce and consume factorizations in the shared format; every backend packs identically, so a
 * decomposition from one backend solves correctly on another. [lu] and [piv] are live buffers, not
 * copies — treat them as read-only.
 *
 * @property n the matrix dimension.
 * @property lu the packed `L`\`U` factors, row-major, length `n * n`.
 * @property piv the row permutation.
 * @property singular whether a zero pivot was encountered.
 */
class LuDecomposition(val n: Int, val lu: DoubleArray, val piv: IntArray, val singular: Boolean) {
    init {
        require(lu.size == n * n) { "lu length ${lu.size} != ${n * n}" }
        require(piv.size == n) { "piv length ${piv.size} != $n" }
    }
}

/** `det(A)` from the factorization: `sign(P) · ∏ U[k][k]`, or exactly `0.0` when [LuDecomposition.singular].
 *  The floating-point counterpart of [SparseLu.determinant]. */
fun LuDecomposition.determinant(): Double {
    if (singular) return 0.0
    var d = permutationSign(piv)
    for (k in 0 until n) d *= lu[k * n + k]
    return d
}

/** Sign of the permutation [p] (`p[k]` = original index at position `k`): `(-1)^(size − cycles)`. */
internal fun permutationSign(p: IntArray): Double {
    val seen = BooleanArray(p.size)
    var cycles = 0
    for (s in p.indices) {
        if (seen[s]) continue
        cycles++
        var i = s
        while (!seen[i]) {
            seen[i] = true
            i = p[i]
        }
    }
    return if ((p.size - cycles) % 2 == 0) 1.0 else -1.0
}

/**
 * Portable pure-Kotlin backend — correct on every target, no native dependency, and the semantic
 * reference a native backend is validated against. Textbook Doolittle LU with partial pivoting and naive
 * (SIMD-assisted where the level-1 primitives kick in) level-2/3 loops.
 */
object ReferenceLinearAlgebra : LinearAlgebra {
    override val name: String get() = "reference"

    override fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        require(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        require(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        if (beta == 0.0) {
            y.fill(0.0)
        } else if (beta != 1.0) {
            denseScale(y, 0, beta, y.size)
        }
        if (alpha == 0.0) return
        val ad = a.data
        val cols = a.cols
        if (transpose) {
            for (i in 0 until a.rows) {
                val xi = alpha * x[i]
                if (xi != 0.0) denseAxpy(y, 0, xi, ad, i * cols, cols)
            }
        } else {
            for (i in 0 until a.rows) {
                y[i] += alpha * denseDot(ad, i * cols, x, 0, cols)
            }
        }
    }

    @Suppress("LongParameterList", "CyclomaticComplexMethod")
    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val k = if (transposeA) a.rows else a.cols
        val kB = if (transposeB) b.cols else b.rows
        val n = if (transposeB) b.rows else b.cols
        require(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        require(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        val cd = c.data
        if (beta == 0.0) {
            cd.fill(0.0)
        } else if (beta != 1.0) {
            denseScale(cd, 0, beta, cd.size)
        }
        if (alpha == 0.0 || m == 0 || n == 0 || k == 0) return
        if (transposeA && transposeB) {
            // C = alpha·Aᵀ·Bᵀ + C: materialize Aᵀ once, then reuse the (noT, T) row-dot kernel.
            val at = DenseMatrix(m, k)
            for (i in 0 until m) for (p in 0 until k) at[i, p] = a[p, i]
            gemm(alpha, at, transposeA = false, b, transposeB = true, beta = 1.0, c = c)
            return
        }
        val ad = a.data
        val bd = b.data
        when {
            // C += alpha·A·B: rank-1 axpy sweeps, all runs contiguous.
            !transposeA && !transposeB -> for (i in 0 until m) {
                for (p in 0 until k) {
                    val aip = alpha * ad[i * k + p]
                    if (aip != 0.0) denseAxpy(cd, i * n, aip, bd, p * n, n)
                }
            }

            // C += alpha·A·Bᵀ: entry (i, j) is the dot of two contiguous rows.
            !transposeA -> for (i in 0 until m) {
                for (j in 0 until n) {
                    cd[i * n + j] += alpha * denseDot(ad, i * k, bd, j * k, k)
                }
            }

            // C += alpha·Aᵀ·B: p outer keeps both the A read row and the B axpy row contiguous.
            else -> for (p in 0 until k) {
                for (i in 0 until m) {
                    val api = alpha * ad[p * m + i]
                    if (api != 0.0) denseAxpy(cd, i * n, api, bd, p * n, n)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature
    override fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, uplo: Uplo) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        require(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        val cd = c.data
        scaleUplo(cd, n, beta, uplo)
        if (alpha == 0.0 || n == 0 || k == 0) return
        val ad = a.data
        if (!transpose) {
            // C += alpha·A·Aᵀ: entry (i, j) is the dot of contiguous rows i and j; computed once per
            // pair, then written to the selected triangle(s).
            for (i in 0 until n) {
                for (j in 0..i) {
                    val v = alpha * denseDot(ad, i * k, ad, j * k, k)
                    addUplo(cd, n, i, j, v, uplo)
                }
            }
        } else {
            // C += alpha·Aᵀ·A = alpha·Σₚ A[p,:]ᵀ·A[p,:]: rank-1 sweeps over contiguous rows of A into
            // a scratch buffer, then written per triangle — the sweeps alone are not bit-symmetric
            // ((alpha·x)·y and (alpha·y)·x round differently), and the contract promises an exactly
            // symmetric alpha term.
            val w = DoubleArray(n * n)
            for (p in 0 until k) {
                val base = p * n
                for (i in 0 until n) {
                    val f = alpha * ad[base + i]
                    if (f != 0.0) denseAxpy(w, i * n, f, ad, base, n)
                }
            }
            for (i in 0 until n) {
                for (j in 0..i) {
                    addUplo(cd, n, i, j, w[i * n + j], uplo)
                }
            }
        }
    }

    /** Add the symmetric term `v` for the pair `(i, j)`, `j <= i`, into the triangle(s) [uplo] selects. */
    @Suppress("LongParameterList")
    private fun addUplo(cd: DoubleArray, n: Int, i: Int, j: Int, v: Double, uplo: Uplo) {
        if (uplo != Uplo.UPPER) {
            cd[i * n + j] += v
        } else if (i == j) {
            cd[i * n + i] += v // the diagonal belongs to both triangles
        }
        if (uplo != Uplo.LOWER && i != j) cd[j * n + i] += v
    }

    /** `beta` scale of the region [uplo] selects, honoring the `beta == 0` overwrite convention. */
    private fun scaleUplo(cd: DoubleArray, n: Int, beta: Double, uplo: Uplo) {
        if (uplo == Uplo.FULL) {
            if (beta == 0.0) {
                cd.fill(0.0)
            } else if (beta != 1.0) {
                denseScale(cd, 0, beta, cd.size)
            }
            return
        }
        if (beta == 1.0) return
        for (i in 0 until n) {
            val from = if (uplo == Uplo.LOWER) i * n else i * n + i
            val len = if (uplo == Uplo.LOWER) i + 1 else n - i
            if (beta == 0.0) cd.fill(0.0, from, from + len) else denseScale(cd, from, beta, len)
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymv signature
    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        require(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        require(x.size == n) { "symv: x length ${x.size} != $n" }
        require(y.size == n) { "symv: y length ${y.size} != $n" }
        if (beta == 0.0) {
            y.fill(0.0)
        } else if (beta != 1.0) {
            denseScale(y, 0, beta, n)
        }
        if (alpha == 0.0) return
        val ad = a.data
        // Row i of the stored triangle serves both A[i, j] (a contiguous dot) and its mirror A[j, i]
        // (a contiguous axpy), so symmetry never forces a strided column walk.
        for (i in 0 until n) {
            val base = i * n
            val xi = alpha * x[i]
            if (lower) {
                y[i] += alpha * denseDot(ad, base, x, 0, i) + xi * ad[base + i]
                if (xi != 0.0) denseAxpy(y, 0, xi, ad, base, i)
            } else {
                y[i] += alpha * denseDot(ad, base + i + 1, x, i + 1, n - i - 1) + xi * ad[base + i]
                if (xi != 0.0) denseAxpy(y, i + 1, xi, ad, base + i + 1, n - i - 1)
            }
        }
    }

    @Suppress("LongParameterList", "CyclomaticComplexMethod") // the BLAS dsymm signature
    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        require(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
        val m = a.rows
        require(c.rows == b.rows && c.cols == b.cols) {
            "symm: C is ${c.rows}x${c.cols} but B is ${b.rows}x${b.cols}"
        }
        if (right) {
            require(b.cols == m) { "symm right: B has ${b.cols} cols, expected $m" }
            val cd = c.data
            if (beta == 0.0) {
                cd.fill(0.0)
            } else if (beta != 1.0) {
                denseScale(cd, 0, beta, cd.size)
            }
            if (alpha == 0.0 || m == 0 || b.rows == 0) return
            // Row r of the product is (A · B[r,:]ᵀ)ᵀ by symmetry: a contiguous per-row symv.
            val x = DoubleArray(m)
            val y = DoubleArray(m)
            for (r in 0 until b.rows) {
                b.data.copyInto(x, 0, r * m, (r + 1) * m)
                y.fill(0.0)
                symv(alpha, a, x, 0.0, y, lower)
                denseAxpy(cd, r * m, 1.0, y, 0, m)
            }
            return
        }
        val p = b.cols
        require(b.rows == m) { "symm: B has ${b.rows} rows, expected $m" }
        val cd = c.data
        if (beta == 0.0) {
            cd.fill(0.0)
        } else if (beta != 1.0) {
            denseScale(cd, 0, beta, cd.size)
        }
        if (alpha == 0.0 || m == 0 || p == 0) return
        val ad = a.data
        val bd = b.data
        for (i in 0 until m) {
            val base = i * m
            val js = if (lower) 0..i else i until m
            for (j in js) {
                val aij = alpha * ad[base + j]
                if (aij != 0.0) {
                    denseAxpy(cd, i * p, aij, bd, j * p, p)
                    if (j != i) denseAxpy(cd, j * p, aij, bd, i * p, p)
                }
            }
        }
    }

    @Suppress("CyclomaticComplexMethod", "LongMethod") // netlib dsytf2's control flow, kept recognizable
    override fun ldl(a: DenseMatrix): LdlDecomposition {
        require(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val w = a.data.copyOf()
        val ipiv = IntArray(n)
        var singular = false
        val colK = DoubleArray(n)
        val colK1 = DoubleArray(n)
        var k = 0
        while (k < n) {
            var kstep = 1
            val absakk = abs(w[k * n + k])
            var imax = k
            var colmax = 0.0
            for (i in k + 1 until n) {
                val v = abs(w[i * n + k])
                if (v > colmax) {
                    colmax = v
                    imax = i
                }
            }
            if (maxOf(absakk, colmax) == 0.0) {
                // Zero pivot column: record and move on without eliminating, as dsytf2 does.
                singular = true
                ipiv[k] = k + 1
                k += 1
                continue
            }
            val kp: Int
            if (absakk >= BUNCH_KAUFMAN_ALPHA * colmax) {
                kp = k
            } else {
                var rowmax = 0.0
                for (j in k until imax) rowmax = maxOf(rowmax, abs(w[imax * n + j]))
                for (j in imax + 1 until n) rowmax = maxOf(rowmax, abs(w[j * n + imax]))
                if (absakk * rowmax >= BUNCH_KAUFMAN_ALPHA * colmax * colmax) {
                    kp = k
                } else if (abs(w[imax * n + imax]) >= BUNCH_KAUFMAN_ALPHA * rowmax) {
                    kp = imax
                } else {
                    kp = imax
                    kstep = 2
                }
            }
            val kk = k + kstep - 1
            if (kp != kk) {
                // Interchange rows/columns kk and kp of the trailing submatrix, dsytf2-style: the
                // already-computed L columns to the left stay put; solve replays ipiv instead.
                for (i in kp + 1 until n) {
                    val t = w[i * n + kk]
                    w[i * n + kk] = w[i * n + kp]
                    w[i * n + kp] = t
                }
                for (j in kk + 1 until kp) {
                    val t = w[j * n + kk]
                    w[j * n + kk] = w[kp * n + j]
                    w[kp * n + j] = t
                }
                val t = w[kk * n + kk]
                w[kk * n + kk] = w[kp * n + kp]
                w[kp * n + kp] = t
                if (kstep == 2) {
                    val s = w[(k + 1) * n + k]
                    w[(k + 1) * n + k] = w[kp * n + k]
                    w[kp * n + k] = s
                }
            }
            if (kstep == 1) {
                // A[k+1.., k+1..] -= d11·col·colᵀ with col the raw pivot column, then scale the column
                // into multipliers. A contiguous copy of the column turns every row update into an axpy.
                val d11 = 1.0 / w[k * n + k]
                for (i in k + 1 until n) colK[i] = w[i * n + k]
                for (i in k + 1 until n) {
                    val f = -d11 * colK[i]
                    if (f != 0.0) denseAxpy(w, i * n + k + 1, f, colK, k + 1, i - k)
                }
                for (i in k + 1 until n) w[i * n + k] = colK[i] * d11
                ipiv[k] = kp + 1
            } else {
                if (k < n - 2) {
                    // 2×2 pivot block: netlib's d21-scaled inverse, applied row-wise via two axpys.
                    val d21 = w[(k + 1) * n + k]
                    val d11 = w[(k + 1) * n + k + 1] / d21
                    val d22 = w[k * n + k] / d21
                    val t = 1.0 / (d11 * d22 - 1.0)
                    val d21inv = t / d21
                    for (j in k + 2 until n) {
                        val cj = w[j * n + k]
                        val cj1 = w[j * n + k + 1]
                        colK[j] = d21inv * (d11 * cj - cj1)
                        colK1[j] = d21inv * (d22 * cj1 - cj)
                    }
                    for (i in k + 2 until n) {
                        val ci = w[i * n + k]
                        val ci1 = w[i * n + k + 1]
                        if (ci != 0.0) denseAxpy(w, i * n + k + 2, -ci, colK, k + 2, i - k - 1)
                        if (ci1 != 0.0) denseAxpy(w, i * n + k + 2, -ci1, colK1, k + 2, i - k - 1)
                    }
                    for (i in k + 2 until n) {
                        w[i * n + k] = colK[i]
                        w[i * n + k + 1] = colK1[i]
                    }
                }
                ipiv[k] = -(kp + 1)
                ipiv[k + 1] = -(kp + 1)
            }
            k += kstep
        }
        return LdlDecomposition(n, w, ipiv, singular)
    }

    override fun solve(ldl: LdlDecomposition, b: DoubleArray): DoubleArray {
        val n = ldl.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        val w = ldl.ldl
        val ipiv = ldl.ipiv
        val x = b.copyOf()
        // Forward: replay interchanges, apply L block columns, divide by the D blocks (dsytrs lower).
        var k = 0
        while (k < n) {
            if (ipiv[k] > 0) {
                val kp = ipiv[k] - 1
                if (kp != k) {
                    val t = x[k]
                    x[k] = x[kp]
                    x[kp] = t
                }
                val xk = x[k]
                if (xk != 0.0) for (i in k + 1 until n) x[i] -= w[i * n + k] * xk
                x[k] = xk / w[k * n + k]
                k += 1
            } else {
                val kp = -ipiv[k] - 1
                if (kp != k + 1) {
                    val t = x[k + 1]
                    x[k + 1] = x[kp]
                    x[kp] = t
                }
                val xk = x[k]
                val xk1 = x[k + 1]
                for (i in k + 2 until n) x[i] -= w[i * n + k] * xk + w[i * n + k + 1] * xk1
                val akm1k = w[(k + 1) * n + k]
                val akm1 = w[k * n + k] / akm1k
                val ak = w[(k + 1) * n + k + 1] / akm1k
                val denom = akm1 * ak - 1.0
                val bkm1 = xk / akm1k
                val bk = xk1 / akm1k
                x[k] = (ak * bkm1 - bk) / denom
                x[k + 1] = (akm1 * bk - bkm1) / denom
                k += 2
            }
        }
        // Backward: apply Lᵀ block rows and undo the interchanges in reverse.
        k = n - 1
        while (k >= 0) {
            if (ipiv[k] > 0) {
                var s = x[k]
                for (i in k + 1 until n) s -= w[i * n + k] * x[i]
                x[k] = s
                val kp = ipiv[k] - 1
                if (kp != k) {
                    val t = x[k]
                    x[k] = x[kp]
                    x[kp] = t
                }
                k -= 1
            } else {
                val k0 = k - 1
                var s0 = x[k0]
                var s1 = x[k]
                for (i in k + 1 until n) {
                    s0 -= w[i * n + k0] * x[i]
                    s1 -= w[i * n + k] * x[i]
                }
                x[k0] = s0
                x[k] = s1
                val kp = -ipiv[k] - 1
                if (kp != k) {
                    val t = x[k]
                    x[k] = x[kp]
                    x[kp] = t
                }
                k -= 2
            }
        }
        return x
    }

    override fun qr(a: DenseMatrix): QrDecomposition {
        val m = a.rows
        val n = a.cols
        val k = minOf(m, n)
        val buf = a.data.copyOf()
        val tau = DoubleArray(k)
        val w = DoubleArray(if (n > 0) n - 1 else 0)
        for (col in 0 until k) {
            tau[col] = householderColumn(buf, m, n, col)
            if (tau[col] == 0.0) continue
            // Apply H = I − tau·v·vᵀ to the trailing columns, row-oriented so every run is contiguous:
            // w = vᵀ·A(sub) accumulates scaled row segments, then each row gets a −tau·v_i·w update.
            val width = n - col - 1
            if (width == 0) continue
            w.fill(0.0, 0, width)
            for (i in col until m) {
                val vi = if (i == col) 1.0 else buf[i * n + col]
                if (vi != 0.0) denseAxpy(w, 0, vi, buf, i * n + col + 1, width)
            }
            for (i in col until m) {
                val vi = if (i == col) 1.0 else buf[i * n + col]
                val f = -tau[col] * vi
                if (f != 0.0) denseAxpy(buf, i * n + col + 1, f, w, 0, width)
            }
        }
        return QrDecomposition(m, n, buf, tau)
    }

    /** Build the Householder reflector for [col] in place (LAPACK `dlarfg`): returns `tau`, stores the
     *  scaled vector below the diagonal and `beta` (the new `R` diagonal) on it. `tau == 0` means the
     *  column was already zero below the diagonal position and `H = I`. */
    private fun householderColumn(buf: DoubleArray, m: Int, n: Int, col: Int): Double {
        var normSq = 0.0
        for (i in col until m) {
            val v = buf[i * n + col]
            normSq += v * v
        }
        if (normSq == 0.0) return 0.0
        val alpha = buf[col * n + col]
        val norm = sqrt(normSq)
        val beta = if (alpha >= 0.0) -norm else norm
        val v0 = alpha - beta
        for (i in col + 1 until m) buf[i * n + col] /= v0
        buf[col * n + col] = beta
        return (beta - alpha) / beta
    }

    override fun applyQ(qr: QrDecomposition, y: DoubleArray, transpose: Boolean): DoubleArray {
        val m = qr.m
        val n = qr.n
        require(y.size == m) { "applyQ: y length ${y.size} != $m" }
        val x = y.copyOf()
        val k = qr.tau.size
        val buf = qr.qr
        // Q = H_0·H_1···H_{k−1} and each H is symmetric, so Qᵀ applies them in ascending order and
        // Q in descending.
        val order = if (transpose) 0 until k else k - 1 downTo 0
        for (col in order) {
            val t = qr.tau[col]
            if (t == 0.0) continue
            var s = x[col]
            for (i in col + 1 until m) s += buf[i * n + col] * x[i]
            val f = t * s
            x[col] -= f
            for (i in col + 1 until m) x[i] -= f * buf[i * n + col]
        }
        return x
    }

    override fun factor(a: DenseMatrix): LuDecomposition {
        require(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val lu = a.data.copyOf()
        val piv = IntArray(n) { it }
        var singular = false
        for (k in 0 until n) {
            var p = k
            var max = abs(lu[k * n + k])
            for (i in k + 1 until n) {
                val v = abs(lu[i * n + k])
                if (v > max) {
                    max = v
                    p = i
                }
            }
            if (max == 0.0) {
                singular = true
                continue
            }
            if (p != k) {
                for (j in 0 until n) {
                    val t = lu[k * n + j]
                    lu[k * n + j] = lu[p * n + j]
                    lu[p * n + j] = t
                }
                val tp = piv[k]
                piv[k] = piv[p]
                piv[p] = tp
            }
            val pivot = lu[k * n + k]
            for (i in k + 1 until n) {
                val f = lu[i * n + k] / pivot
                lu[i * n + k] = f
                denseAxpy(lu, i * n + k + 1, -f, lu, k * n + k + 1, n - k - 1)
            }
        }
        return LuDecomposition(n, lu, piv, singular)
    }

    override fun solve(lu: LuDecomposition, b: DoubleArray, transpose: Boolean): DoubleArray {
        val n = lu.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        val a = lu.lu
        val piv = lu.piv
        return if (transpose) solveTranspose(n, a, piv, b) else solveNormal(n, a, piv, b)
    }

    // A x = b, with P A = L U: permute b by P, forward-solve unit-lower L, back-solve U.
    private fun solveNormal(n: Int, a: DoubleArray, piv: IntArray, b: DoubleArray): DoubleArray {
        val x = DoubleArray(n) { b[piv[it]] }
        trsvCore(a, n, x, lower = true, transpose = false, unitDiag = true)
        trsvCore(a, n, x, lower = false, transpose = false, unitDiag = false)
        return x
    }

    // Aᵀ x = b, with Aᵀ = Uᵀ Lᵀ P: forward-solve Uᵀ (lower), back-solve unit-upper Lᵀ, then un-permute.
    private fun solveTranspose(n: Int, a: DoubleArray, piv: IntArray, b: DoubleArray): DoubleArray {
        val y = b.copyOf()
        trsvCore(a, n, y, lower = false, transpose = true, unitDiag = false)
        trsvCore(a, n, y, lower = true, transpose = true, unitDiag = true)
        val x = DoubleArray(n) // undo the row permutation: x[piv[i]] = z[i]
        for (i in 0 until n) x[piv[i]] = y[i]
        return x
    }
}

/**
 * The best backend the current platform provides, or null when only the portable reference is available.
 * On the JVM this discovers providers via `ServiceLoader`, so an optional backend artifact (such as
 * koblas-openblas) activates itself when present on the classpath. Other targets have no runtime
 * discovery and return null; a backend artifact there is activated explicitly through
 * [installLinearAlgebra].
 */
expect fun platformLinearAlgebra(): LinearAlgebra?

private val platformDefault: LinearAlgebra by lazy { platformLinearAlgebra() ?: ReferenceLinearAlgebra }

@Volatile
private var installed: LinearAlgebra? = null

@Volatile
private var registered: LinearAlgebra? = null

/** The active linear-algebra backend: an [installLinearAlgebra] override when set, else the
 *  best [registerLinearAlgebra] candidate, else the [platformLinearAlgebra] backend when present,
 *  else the portable [ReferenceLinearAlgebra]. */
val koblas: LinearAlgebra get() = installed ?: registered ?: platformDefault

/** What this runtime resolved on both performance seams, for startup logging — e.g.
 *  `"backend=openblas, primitives=simd(8 lanes)"` ([koblas].name and [mathBackend]). */
val koblasInfo: String get() = "backend=${koblas.name}, primitives=$mathBackend"

/**
 * Overrides the backend [koblas] resolves to, taking precedence over every automatic mechanism —
 * registration and platform discovery alike. Passing null restores automatic selection. The switch
 * is visible to subsequent [koblas] reads but is not synchronized with operations in flight.
 */
fun installLinearAlgebra(backend: LinearAlgebra?) {
    installed = backend
}

/**
 * Offers [backend] for automatic selection: it becomes active only while no [installLinearAlgebra]
 * override is set and no previously registered backend has a higher [LinearAlgebra.priority]. This
 * is how backend artifacts activate themselves on platforms without classpath discovery — the
 * koblas-cblas startup registration goes through here — and it keeps the strongest of several
 * candidates: openblas over cblas over the reference.
 */
fun registerLinearAlgebra(backend: LinearAlgebra) {
    val current = registered
    if (current == null || backend.priority > current.priority) registered = backend
}

/** Test hook: clears [registerLinearAlgebra] state so selection tests are order-independent. */
internal fun resetRegisteredLinearAlgebra() {
    registered = null
}

/** LU-factorize this square matrix with the active backend ([koblas]). */
fun DenseMatrix.lu(): LuDecomposition = koblas.factor(this)

/** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for this factorization with the active backend. */
fun LuDecomposition.solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.solve(this, b, transpose)

/** Matrix-matrix product `this · other` with the active backend. */
fun DenseMatrix.matMul(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)
