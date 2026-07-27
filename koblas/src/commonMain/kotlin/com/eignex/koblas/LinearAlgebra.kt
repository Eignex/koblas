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

    /** Matrix-matrix product `A · B` (BLAS `dgemm`); `A.cols` must equal `B.rows`. */
    fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix

    /** LU factorization with partial pivoting of a square [a] (LAPACK `dgetrf`); [a] is not modified. */
    fun factor(a: DenseMatrix): LuDecomposition

    /** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for the factorization [lu] (LAPACK `dgetrs`);
     *  returns a fresh `x`. [transpose] serves the simplex's BTRAN (`Bᵀ y = c`) against a factored basis. */
    fun solve(lu: LuDecomposition, b: DoubleArray, transpose: Boolean = false): DoubleArray
}

/**
 * A general LU factorization with partial pivoting: `P·A = L·U`, the unit-lower `L` and upper `U` packed
 * into one flat row-major [lu] buffer (`L` below the diagonal, `U` on and above) and the row permutation
 * in [piv] (`piv[k]` is the original row now at position `k`). [singular] is set when a zero pivot was
 * hit; [LinearAlgebra.solve] on a singular factorization is not meaningful. Produced by
 * [LinearAlgebra.factor].
 *
 * @property n the matrix dimension.
 * @property lu the packed `L`\`U` factors, row-major, length `n * n`.
 * @property piv the row permutation.
 * @property singular whether a zero pivot was encountered.
 */
class LuDecomposition internal constructor(
    val n: Int,
    internal val lu: DoubleArray,
    internal val piv: IntArray,
    val singular: Boolean,
)

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

    override fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        require(a.cols == b.rows) { "gemm: ${a.rows}x${a.cols} · ${b.rows}x${b.cols}" }
        val m = a.rows
        val k = a.cols
        val n = b.cols
        val c = DoubleArray(m * n)
        val ad = a.data
        val bd = b.data
        for (i in 0 until m) {
            val aBase = i * k
            val cBase = i * n
            for (p in 0 until k) {
                val aip = ad[aBase + p]
                if (aip != 0.0) denseAxpy(c, cBase, aip, bd, p * n, n)
            }
        }
        return DenseMatrix(m, n, c)
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
    // Both substitutions read contiguous row runs of the packed factors, so they reduce to [denseDot].
    private fun solveNormal(n: Int, a: DoubleArray, piv: IntArray, b: DoubleArray): DoubleArray {
        val x = DoubleArray(n) { b[piv[it]] }
        for (i in 0 until n) {
            x[i] -= denseDot(a, i * n, x, 0, i)
        }
        for (i in n - 1 downTo 0) {
            val base = i * n
            x[i] = (x[i] - denseDot(a, base + i + 1, x, i + 1, n - i - 1)) / a[base + i]
        }
        return x
    }

    // Aᵀ x = b, with Aᵀ = Uᵀ Lᵀ P: forward-solve Uᵀ (lower), back-solve unit-upper Lᵀ, then un-permute.
    // The transposed factors are columns of the row-major packing, so the row-oriented form would walk
    // strided; instead each substitution is column-oriented — once y[i] is final, its contribution is
    // subtracted from the remaining right-hand side along contiguous row i via [denseAxpy].
    private fun solveTranspose(n: Int, a: DoubleArray, piv: IntArray, b: DoubleArray): DoubleArray {
        val y = b.copyOf()
        for (i in 0 until n) { // Uᵀ y = b (Uᵀ lower-triangular, diagonal from U)
            val yi = y[i] / a[i * n + i]
            y[i] = yi
            if (yi != 0.0) denseAxpy(y, i + 1, -yi, a, i * n + i + 1, n - i - 1)
        }
        for (i in n - 1 downTo 0) { // Lᵀ z = y (Lᵀ unit-upper)
            val zi = y[i]
            if (zi != 0.0) denseAxpy(y, 0, -zi, a, i * n, i)
        }
        val x = DoubleArray(n) // undo the row permutation: x[piv[i]] = z[i]
        for (i in 0 until n) x[piv[i]] = y[i]
        return x
    }
}

/**
 * The best backend the current platform provides, or null when only the portable reference is available.
 * A platform's `actual` returns a native BLAS/LAPACK-backed [LinearAlgebra] once one is wired; today every
 * target returns null, so [koblas] resolves to [ReferenceLinearAlgebra].
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
