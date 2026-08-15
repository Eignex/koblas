@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.requireShape

/**
 * The portable dense matrix routines, the semantic reference a native [Blas] is validated against.
 *
 * @param kernels the vector kernels the inner loops use, or null to follow the [KoblasContext] default.
 */
internal class ReferenceBlas(private val kernels: VectorKernels? = null) : Blas {
    override val name: String get() = "reference"

    override val isPortable: Boolean get() = true

    /** These routines' kernels, or the process default when they were given none. */
    override val vectorKernels: VectorKernels get() = kernels ?: koblas.vectorKernels

    override fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        applyBeta(vectorKernels, y, 0, y.size, beta)
        if (alpha == 0.0) return
        val ad = a.data
        val rows = a.rows
        if (!transpose) {
            for (j in 0 until a.cols) {
                val xj = alpha * x[j]
                if (xj != 0.0) vectorKernels.axpy(y, 0, xj, ad, j * rows, rows)
            }
        } else {
            val quads = DoubleArray(4)
            var j = 0
            val bound = a.cols - 3
            while (j < bound) {
                vectorKernels.dot4(ad, j * rows, rows, x, 0, rows, quads, 0)
                y[j] += alpha * quads[0]
                y[j + 1] += alpha * quads[1]
                y[j + 2] += alpha * quads[2]
                y[j + 3] += alpha * quads[3]
                j += 4
            }
            while (j < a.cols) {
                y[j] += alpha * vectorKernels.dot(ad, j * rows, x, 0, rows)
                j++
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
        requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
        requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
        val cd = c.data
        applyBeta(vectorKernels, cd, 0, cd.size, beta)
        if (alpha == 0.0 || m == 0 || n == 0 || k == 0) return
        val ad = a.data
        val bd = b.data
        val kernels = vectorKernels
        if (transposeA && transposeB) {
            // With both transposed one operand is strided whichever way the loops run, so the smaller one
            // is transposed first. That copy is O(k·n) or O(k·m) against the product's O(m·k·n).
            if (b.rows * b.cols <= a.rows * a.cols) {
                val bt = DoubleArray(k * n)
                for (j in 0 until n) for (p in 0 until k) bt[p + j * k] = bd[j + p * n]
                gemm(alpha, a, true, DenseMatrix.wrap(k, n, bt), false, 1.0, c)
            } else {
                val at = DoubleArray(m * k)
                for (i in 0 until m) for (p in 0 until k) at[i + p * m] = ad[p + i * k]
                gemm(alpha, DenseMatrix.wrap(m, k, at), false, b, true, 1.0, c)
            }
            return
        }
        when {
            transposeA -> {
                val quads = DoubleArray(4)
                for (j in 0 until n) {
                    var i = 0
                    val bound = m - 3
                    while (i < bound) {
                        kernels.dot4(ad, i * k, k, bd, j * k, k, quads, 0)
                        cd[i + j * m] += alpha * quads[0]
                        cd[i + 1 + j * m] += alpha * quads[1]
                        cd[i + 2 + j * m] += alpha * quads[2]
                        cd[i + 3 + j * m] += alpha * quads[3]
                        i += 4
                    }
                    while (i < m) {
                        cd[i + j * m] += alpha * kernels.dot(ad, i * k, bd, j * k, k)
                        i++
                    }
                }
            }

            !transposeB -> for (j in 0 until n) {
                for (p in 0 until k) {
                    val bpj = alpha * bd[p + j * k]
                    if (bpj != 0.0) kernels.axpy(cd, j * m, bpj, ad, p * m, m)
                }
            }

            else -> for (j in 0 until n) {
                for (p in 0 until k) {
                    val bjp = alpha * bd[j + p * n]
                    if (bjp != 0.0) kernels.axpy(cd, j * m, bjp, ad, p * m, m)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus optional scratch
    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        uplo: Uplo,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        requireShape(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
        val cd = c.data
        scaleUplo(vectorKernels, cd, n, beta, uplo)
        if (alpha == 0.0 || n == 0 || k == 0) return
        val ad = a.data
        // The dot form holds each output entry in a register, where accumulating outer products would
        // stream all of C once per column of A. So a non-transposed A is transposed first, an O(n·k) copy
        // against the product's O(n²·k/2), and both orientations run the same loop below.
        val kernels = vectorKernels
        val at = if (transpose) null else workspace?.take(n * k) ?: DoubleArray(n * k)
        if (at != null) {
            for (p in 0 until k) {
                val base = p * n
                for (j in 0 until n) at[p + j * k] = ad[base + j]
            }
        }
        val source = at ?: ad
        for (j in 0 until n) {
            for (i in j until n) {
                val v = alpha * kernels.dot(source, i * k, source, j * k, k)
                addUplo(cd, n, i, j, v, uplo)
            }
        }
        if (at != null) workspace?.release(at)
    }

    @Suppress("LongParameterList") // the BLAS dsymv signature
    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireShape(a.rows == a.cols) { "symv: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        requireShape(x.size == n) { "symv: x length ${x.size} != $n" }
        requireShape(y.size == n) { "symv: y length ${y.size} != $n" }
        applyBeta(vectorKernels, y, 0, n, beta)
        if (alpha == 0.0) return
        symvAccumulate(alpha, a.data, n, x, 0, y, 0, lower)
    }

    /** Accumulates alpha times A times x into y for the symmetric `n×n` [ad], reading only the [lower] or
     *  upper triangle. */
    @Suppress("LongParameterList") // two buffers with offsets plus the triangle flag
    private fun symvAccumulate(
        alpha: Double,
        ad: DoubleArray,
        n: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        lower: Boolean,
    ) {
        for (j in 0 until n) {
            val base = j + j * n
            val xj = alpha * x[xOff + j]
            if (lower) {
                val len = n - j - 1
                y[yOff + j] += alpha * vectorKernels.dot(ad, base + 1, x, xOff + j + 1, len) + xj * ad[base]
                if (xj != 0.0) vectorKernels.axpy(y, yOff + j + 1, xj, ad, base + 1, len)
            } else {
                y[yOff + j] += alpha * vectorKernels.dot(ad, j * n, x, xOff, j) + xj * ad[base]
                if (xj != 0.0) vectorKernels.axpy(y, yOff, xj, ad, j * n, j)
            }
        }
    }

    /** A(i, j) of a symmetric `n×n` matrix stored in its [lower] or upper triangle only. */
    private fun symEntry(ad: DoubleArray, n: Int, i: Int, j: Int, lower: Boolean): Double {
        val hi = if (i > j) i else j
        val lo = if (i > j) j else i
        return if (lower) ad[hi + lo * n] else ad[lo + hi * n]
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
        requireShape(a.rows == a.cols) { "symm: matrix must be square, got ${a.rows}x${a.cols}" }
        val m = a.rows
        requireShape(c.rows == b.rows && c.cols == b.cols) {
            "symm: C is ${c.rows}x${c.cols} but B is ${b.rows}x${b.cols}"
        }
        val cd = c.data
        applyBeta(vectorKernels, cd, 0, cd.size, beta)
        if (right) {
            requireShape(b.cols == m) { "symm right: B has ${b.cols} cols, expected $m" }
            if (alpha == 0.0 || m == 0 || b.rows == 0) return
            val rows = b.rows
            val ad = a.data
            val bd = b.data
            for (j in 0 until m) {
                for (p in 0 until m) {
                    val apj = alpha * symEntry(ad, m, p, j, lower)
                    if (apj != 0.0) vectorKernels.axpy(cd, j * rows, apj, bd, p * rows, rows)
                }
            }
            return
        }
        requireShape(b.rows == m) { "symm: B has ${b.rows} rows, expected $m" }
        if (alpha == 0.0 || m == 0 || b.cols == 0) return
        for (j in 0 until b.cols) {
            symvAccumulate(alpha, a.data, m, b.data, j * m, cd, j * m, lower)
        }
    }
}
