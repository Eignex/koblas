@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas

import kotlin.math.abs

/**
 * The swappable compute backend for the heavier dense operations — level-2/3 BLAS ([gemv], [gemm]) and a
 * general LU [factor]/[solve] — where a native BLAS/LAPACK or GPU implementation is worth dispatching to.
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
     * when [transpose] (BLAS `dsyrk`). Deviation from BLAS: there is no `uplo` parameter — the full
     * symmetric result is produced (the alpha term is applied to both triangles). [c] must be square
     * with dimension `op(A).rows`. Per BLAS convention, `beta == 0.0` overwrites [c] without reading
     * it, and `alpha == 0.0` reduces to the `beta` scale.
     */
    fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix)

    /** LU factorization with partial pivoting of a square [a] (LAPACK `dgetrf`); [a] is not modified. */
    fun factor(a: DenseMatrix): LuDecomposition

    /** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for the factorization [lu] (LAPACK `dgetrs`);
     *  returns a fresh `x`. [transpose] serves the simplex's BTRAN (`Bᵀ y = c`) against a factored basis. */
    fun solve(lu: LuDecomposition, b: DoubleArray, transpose: Boolean = false): DoubleArray

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

    override fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        require(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        val cd = c.data
        if (beta == 0.0) {
            cd.fill(0.0)
        } else if (beta != 1.0) {
            denseScale(cd, 0, beta, cd.size)
        }
        if (alpha == 0.0 || n == 0 || k == 0) return
        val ad = a.data
        if (!transpose) {
            // C += alpha·A·Aᵀ: entry (i, j) is the dot of contiguous rows i and j; computed once for
            // the lower triangle and added to both mirror slots.
            for (i in 0 until n) {
                for (j in 0..i) {
                    val v = alpha * denseDot(ad, i * k, ad, j * k, k)
                    cd[i * n + j] += v
                    if (i != j) cd[j * n + i] += v
                }
            }
        } else {
            // C += alpha·Aᵀ·A = alpha·Σₚ A[p,:]ᵀ·A[p,:]: rank-1 sweeps over contiguous rows of A into
            // a scratch buffer, whose lower triangle is then mirrored into C — the sweeps alone are
            // not bit-symmetric ((alpha·x)·y and (alpha·y)·x round differently), and the contract
            // promises an exactly symmetric alpha term.
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
                    val v = w[i * n + j]
                    cd[i * n + j] += v
                    if (i != j) cd[j * n + i] += v
                }
            }
        }
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
 * koblas-openblas) activates itself when present on the classpath; all other targets return null today,
 * resolving [koblas] to [ReferenceLinearAlgebra].
 */
expect fun platformLinearAlgebra(): LinearAlgebra?

/** The active linear-algebra backend: the [platformLinearAlgebra] native backend when present, else the
 *  portable [ReferenceLinearAlgebra]. */
val koblas: LinearAlgebra = platformLinearAlgebra() ?: ReferenceLinearAlgebra

/** LU-factorize this square matrix with the active backend ([koblas]). */
fun DenseMatrix.lu(): LuDecomposition = koblas.factor(this)

/** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for this factorization with the active backend. */
fun LuDecomposition.solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.solve(this, b, transpose)

/** Matrix-matrix product `this · other` with the active backend. */
fun DenseMatrix.matMul(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)
