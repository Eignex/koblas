package com.eignex.koblas

import kotlin.math.abs

/**
 * The dense linear-algebra operations koblas dispatches to a backend: BLAS-1 ([dot]/[axpy]/[scal]),
 * BLAS-2/3 ([gemv]/[gemm]) and an LU [factor]/[solve] pair. The set is deliberately the minimum a dense
 * linear solver (revised simplex, least squares) needs — it grows only as consumers demand.
 *
 * Implementations range from the always-available pure-Kotlin [ReferenceLinearAlgebra] to a
 * platform-native BLAS/LAPACK binding (see [platformLinearAlgebra]); [koblas] picks the best available.
 * Every operation is over plain [DoubleArray]/[Matrix] buffers so a native backend can pass them across
 * the FFI boundary without repacking.
 */
interface LinearAlgebra {
    /** A short identifier for the backend (e.g. `"reference"`), for diagnostics. */
    val name: String

    /** Dot product `x · y`; the arrays must be the same length. */
    fun dot(x: DoubleArray, y: DoubleArray): Double

    /** `y += alpha * x`, elementwise (BLAS `daxpy`); the arrays must be the same length. */
    fun axpy(alpha: Double, x: DoubleArray, y: DoubleArray)

    /** `x *= alpha`, elementwise (BLAS `dscal`). */
    fun scal(alpha: Double, x: DoubleArray)

    /** Matrix-vector product `A · x`, or `Aᵀ · x` when [transpose] (BLAS `dgemv`). */
    fun gemv(a: Matrix, x: DoubleArray, transpose: Boolean = false): DoubleArray

    /** Matrix-matrix product `A · B` (BLAS `dgemm`); `A.cols` must equal `B.rows`. */
    fun gemm(a: Matrix, b: Matrix): Matrix

    /** LU factorization with partial pivoting of a square matrix [a] (LAPACK `dgetrf`); [a] is copied. */
    fun factor(a: Matrix): LuDecomposition

    /** Solve `A · x = b` for the factorization [lu] (LAPACK `dgetrs`); returns a fresh `x`. */
    fun solve(lu: LuDecomposition, b: DoubleArray): DoubleArray
}

/**
 * An LU factorization with partial pivoting: `P·A = L·U`, with the unit-lower `L` and upper `U` packed
 * into one row-major [lu] buffer and the row permutation in [piv] (`piv[k]` is the original row now at
 * position `k`). [singular] is set when a zero pivot was hit; [LinearAlgebra.solve] on a singular
 * factorization is not meaningful. Produced by [LinearAlgebra.factor].
 *
 * @property n the matrix dimension.
 * @property lu the combined unit-lower `L` and upper `U` factors, row-major.
 * @property piv the row permutation (`piv[k]` is the original row now at position `k`).
 * @property singular whether a zero pivot was encountered.
 */
class LuDecomposition internal constructor(
    val n: Int,
    internal val lu: DoubleArray,
    internal val piv: IntArray,
    val singular: Boolean,
)

/**
 * Portable pure-Kotlin reference backend — correct on every target, no native dependency. Textbook
 * Doolittle LU with partial pivoting and naive BLAS loops; a native backend supersedes it for speed
 * where available, but it is the semantic reference the backends are tested against.
 */
object ReferenceLinearAlgebra : LinearAlgebra {
    override val name: String get() = "reference"

    override fun dot(x: DoubleArray, y: DoubleArray): Double {
        require(x.size == y.size) { "length mismatch ${x.size} != ${y.size}" }
        var acc = 0.0
        for (i in x.indices) acc += x[i] * y[i]
        return acc
    }

    override fun axpy(alpha: Double, x: DoubleArray, y: DoubleArray) {
        require(x.size == y.size) { "length mismatch ${x.size} != ${y.size}" }
        for (i in x.indices) y[i] += alpha * x[i]
    }

    override fun scal(alpha: Double, x: DoubleArray) {
        for (i in x.indices) x[i] *= alpha
    }

    override fun gemv(a: Matrix, x: DoubleArray, transpose: Boolean): DoubleArray {
        if (transpose) {
            require(x.size == a.rows) { "gemvᵀ: x length ${x.size} != rows ${a.rows}" }
            val out = DoubleArray(a.cols)
            for (i in 0 until a.rows) {
                val xi = x[i]
                if (xi == 0.0) continue
                val base = i * a.cols
                for (j in 0 until a.cols) out[j] += a.data[base + j] * xi
            }
            return out
        }
        require(x.size == a.cols) { "gemv: x length ${x.size} != cols ${a.cols}" }
        val out = DoubleArray(a.rows)
        for (i in 0 until a.rows) {
            val base = i * a.cols
            var acc = 0.0
            for (j in 0 until a.cols) acc += a.data[base + j] * x[j]
            out[i] = acc
        }
        return out
    }

    override fun gemm(a: Matrix, b: Matrix): Matrix {
        require(a.cols == b.rows) { "gemm: ${a.rows}x${a.cols} · ${b.rows}x${b.cols}" }
        val m = a.rows
        val k = a.cols
        val n = b.cols
        val c = DoubleArray(m * n)
        for (i in 0 until m) {
            val aBase = i * k
            val cBase = i * n
            for (p in 0 until k) {
                val aip = a.data[aBase + p]
                if (aip == 0.0) continue
                val bBase = p * n
                for (j in 0 until n) c[cBase + j] += aip * b.data[bBase + j]
            }
        }
        return Matrix(m, n, c)
    }

    override fun factor(a: Matrix): LuDecomposition {
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
                for (j in k + 1 until n) lu[i * n + j] -= f * lu[k * n + j]
            }
        }
        return LuDecomposition(n, lu, piv, singular)
    }

    override fun solve(lu: LuDecomposition, b: DoubleArray): DoubleArray {
        val n = lu.n
        require(b.size == n) { "solve: b length ${b.size} != $n" }
        val a = lu.lu
        val piv = lu.piv
        val x = DoubleArray(n) { b[piv[it]] } // apply the row permutation P
        for (i in 0 until n) { // forward substitution: L y = P b, L unit-lower
            var s = x[i]
            val base = i * n
            for (j in 0 until i) s -= a[base + j] * x[j]
            x[i] = s
        }
        for (i in n - 1 downTo 0) { // back substitution: U x = y
            var s = x[i]
            val base = i * n
            for (j in i + 1 until n) s -= a[base + j] * x[j]
            x[i] = s / a[base + i]
        }
        return x
    }
}

/**
 * The best backend the current platform provides, or null when only the portable reference is available.
 * A platform's `actual` returns a native BLAS/LAPACK-backed [LinearAlgebra] once one is wired; today every
 * target returns null, so [koblas] resolves to [ReferenceLinearAlgebra].
 */
expect fun platformLinearAlgebra(): LinearAlgebra?

/** The linear-algebra backend koblas uses: the [platformLinearAlgebra] native backend if present, else
 *  the portable [ReferenceLinearAlgebra]. */
val koblas: LinearAlgebra = platformLinearAlgebra() ?: ReferenceLinearAlgebra
