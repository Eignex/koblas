@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import kotlin.math.abs
import kotlin.math.sqrt

/** Bunch-Kaufman pivot threshold `(1 + sqrt(17)) / 8`, the value minimizing element growth (netlib dsytf2). */
private const val BUNCH_KAUFMAN_ALPHA = 0.6403882032022076

/**
 * The portable dense factorizations, the semantic reference a native [Lapack] is validated against.
 *
 * @param kernels the vector kernels the inner loops use, or null to follow the [KoblasContext] default.
 */
internal class ReferenceLapack(private val kernels: VectorKernels? = null) : Lapack {
    override val name: String get() = "reference"

    /** These routines' kernels, or the process default when they were given none. */
    override val vectorKernels: VectorKernels get() = kernels ?: koblas.vectorKernels

    @Suppress("CyclomaticComplexMethod") // netlib dsytf2's control flow, kept recognizable
    override fun ldl(a: DenseMatrix, workspace: Workspace?): LdlDecomposition {
        requireShape(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        val w = a.data.copyOf()
        val ipiv = IntArray(n)
        val kernels = vectorKernels
        var failedAt = NOT_SINGULAR
        val colK = workspace?.take(n) ?: DoubleArray(n)
        val colK1 = workspace?.take(n) ?: DoubleArray(n)
        var k = 0
        while (k < n) {
            var kstep = 1
            val absakk = abs(w[k + k * n])
            var imax = k
            var colmax = 0.0
            for (i in k + 1 until n) {
                val v = abs(w[i + k * n])
                if (v > colmax) {
                    colmax = v
                    imax = i
                }
            }
            if (maxOf(absakk, colmax) == 0.0) {
                // The first zero pivot is the one reported, matching dsytf2's info.
                if (failedAt == NOT_SINGULAR) failedAt = k
                ipiv[k] = k + 1
                k += 1
                continue
            }
            val kp: Int
            if (absakk >= BUNCH_KAUFMAN_ALPHA * colmax) {
                kp = k
            } else {
                var rowmax = 0.0
                for (j in k until imax) rowmax = maxOf(rowmax, abs(w[imax + j * n]))
                for (j in imax + 1 until n) rowmax = maxOf(rowmax, abs(w[j + imax * n]))
                if (absakk * rowmax >= BUNCH_KAUFMAN_ALPHA * colmax * colmax) {
                    kp = k
                } else if (abs(w[imax + imax * n]) >= BUNCH_KAUFMAN_ALPHA * rowmax) {
                    kp = imax
                } else {
                    kp = imax
                    kstep = 2
                }
            }
            val kk = k + kstep - 1
            if (kp != kk) swapSymmetric(w, n, k, kk, kp, kstep)
            if (kstep == 1) {
                val d11 = 1.0 / w[k + k * n]
                for (j in k + 1 until n) {
                    val f = -d11 * w[j + k * n]
                    if (f != 0.0) kernels.axpy(w, j + j * n, f, w, j + k * n, n - j)
                }
                kernels.scale(w, k + 1 + k * n, d11, n - k - 1)
                ipiv[k] = kp + 1
            } else {
                if (k < n - 2) {
                    val d21 = w[(k + 1) + k * n]
                    val d11 = w[(k + 1) + (k + 1) * n] / d21
                    val d22 = w[k + k * n] / d21
                    val t = 1.0 / (d11 * d22 - 1.0)
                    val d21inv = t / d21
                    for (j in k + 2 until n) {
                        val cj = w[j + k * n]
                        val cj1 = w[j + (k + 1) * n]
                        colK[j] = d21inv * (d11 * cj - cj1)
                        colK1[j] = d21inv * (d22 * cj1 - cj)
                    }
                    for (j in k + 2 until n) {
                        val fj = colK[j]
                        val fj1 = colK1[j]
                        if (fj != 0.0) kernels.axpy(w, j + j * n, -fj, w, j + k * n, n - j)
                        if (fj1 != 0.0) kernels.axpy(w, j + j * n, -fj1, w, j + (k + 1) * n, n - j)
                    }
                    for (i in k + 2 until n) {
                        w[i + k * n] = colK[i]
                        w[i + (k + 1) * n] = colK1[i]
                    }
                }
                ipiv[k] = -(kp + 1)
                ipiv[k + 1] = -(kp + 1)
            }
            k += kstep
        }
        if (workspace != null) {
            workspace.release(colK1)
            workspace.release(colK)
        }
        return LdlDecomposition(n, w, ipiv, failedAt)
    }

    /**
     * Exchange rows and columns [kk] and [kp] of the lower triangle in [w], the symmetric swap `dsytf2`
     * makes once it has chosen a pivot. Pure data movement, no arithmetic.
     */
    @Suppress("LongParameterList") // the pivot pair plus the step width, all of which the caller holds
    private fun swapSymmetric(w: DoubleArray, n: Int, k: Int, kk: Int, kp: Int, kstep: Int) {
        for (i in kp + 1 until n) {
            val t = w[i + kk * n]
            w[i + kk * n] = w[i + kp * n]
            w[i + kp * n] = t
        }
        for (j in kk + 1 until kp) {
            val t = w[j + kk * n]
            w[j + kk * n] = w[kp + j * n]
            w[kp + j * n] = t
        }
        val t = w[kk + kk * n]
        w[kk + kk * n] = w[kp + kp * n]
        w[kp + kp * n] = t
        if (kstep == 2) {
            val s = w[(k + 1) + k * n]
            w[(k + 1) + k * n] = w[kp + k * n]
            w[kp + k * n] = s
        }
    }

    override fun solveInto(ldl: LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray {
        requireFactored(ldl.failedAt, "solve")
        val n = ldl.n
        requireShape(b.size == n) { "solve: b length ${b.size} != $n" }
        requireShape(out.size == n) { "solve: out length ${out.size} != $n" }
        val w = ldl.ldl
        val ipiv = ldl.ipiv
        val x = out
        if (out !== b) b.copyInto(out)
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
                if (xk != 0.0) vectorKernels.axpy(x, k + 1, -xk, w, k + 1 + k * n, n - k - 1)
                x[k] = xk / w[k + k * n]
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
                val len = n - k - 2
                if (xk != 0.0) vectorKernels.axpy(x, k + 2, -xk, w, k + 2 + k * n, len)
                if (xk1 != 0.0) vectorKernels.axpy(x, k + 2, -xk1, w, k + 2 + (k + 1) * n, len)
                val akm1k = w[(k + 1) + k * n]
                val akm1 = w[k + k * n] / akm1k
                val ak = w[(k + 1) + (k + 1) * n] / akm1k
                val denom = akm1 * ak - 1.0
                val bkm1 = xk / akm1k
                val bk = xk1 / akm1k
                x[k] = (ak * bkm1 - bk) / denom
                x[k + 1] = (akm1 * bk - bkm1) / denom
                k += 2
            }
        }
        k = n - 1
        while (k >= 0) {
            if (ipiv[k] > 0) {
                x[k] -= vectorKernels.dot(w, k + 1 + k * n, x, k + 1, n - k - 1)
                val kp = ipiv[k] - 1
                if (kp != k) {
                    val t = x[k]
                    x[k] = x[kp]
                    x[kp] = t
                }
                k -= 1
            } else {
                val k0 = k - 1
                val len = n - k - 1
                val s0 = x[k0] - vectorKernels.dot(w, k + 1 + k0 * n, x, k + 1, len)
                val s1 = x[k] - vectorKernels.dot(w, k + 1 + k * n, x, k + 1, len)
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

    override fun qr(a: DenseMatrix, workspace: Workspace?): QrDecomposition {
        val m = a.rows
        val n = a.cols
        val k = minOf(m, n)
        val buf = a.data.copyOf()
        val tau = DoubleArray(k)
        for (col in 0 until k) {
            tau[col] = householderColumn(buf, m, col)
            applyReflectorToTrailing(buf, m, n, col, tau[col])
        }
        return QrDecomposition(m, n, buf, tau)
    }

    override fun qrPivoted(a: DenseMatrix, tolerance: Double, workspace: Workspace?): PivotedQrDecomposition {
        val m = a.rows
        val n = a.cols
        val k = minOf(m, n)
        val buf = a.data.copyOf()
        val tau = DoubleArray(k)
        val pivots = IntArray(n) { it }
        val current = DoubleArray(n) { c -> vectorKernels.nrm2(buf, c * m, m) }
        val computed = current.copyOf()
        for (col in 0 until k) {
            var best = col
            for (c in col + 1 until n) if (current[c] > current[best]) best = c
            if (best != col) swapColumns(buf, m, pivots, current, computed, col, best)
            tau[col] = householderColumn(buf, m, col)
            applyReflectorToTrailing(buf, m, n, col, tau[col])
            downdateNorms(buf, m, n, col, current, computed)
        }
        return PivotedQrDecomposition(QrDecomposition(m, n, buf, tau), pivots, rankOfPivotedR(buf, m, n, k, tolerance))
    }

    /** Exchange columns [i] and [j] of [buf] and every per-column quantity that tracks them. */
    @Suppress("LongParameterList") // one array per tracked quantity; bundling them would allocate per step
    private fun swapColumns(
        buf: DoubleArray,
        m: Int,
        pivots: IntArray,
        current: DoubleArray,
        computed: DoubleArray,
        i: Int,
        j: Int,
    ) {
        for (r in 0 until m) {
            val t = buf[r + i * m]
            buf[r + i * m] = buf[r + j * m]
            buf[r + j * m] = t
        }
        pivots[i] = pivots[j].also { pivots[j] = pivots[i] }
        current[i] = current[j].also { current[j] = current[i] }
        computed[i] = computed[j].also { computed[j] = computed[i] }
    }

    /** Apply `H = I − tau·v·vᵀ` from column [col] to every column after it. */
    private fun applyReflectorToTrailing(buf: DoubleArray, m: Int, n: Int, col: Int, tau: Double) {
        if (tau == 0.0) return
        val len = m - col - 1
        val vBase = col + 1 + col * m
        for (c in col + 1 until n) {
            val head = col + c * m
            val s = buf[head] + vectorKernels.dot(buf, vBase, buf, head + 1, len)
            val f = -tau * s
            if (f != 0.0) {
                buf[head] += f
                vectorKernels.axpy(buf, head + 1, f, buf, vBase, len)
            }
        }
    }

    /** Shrink each trailing column's norm by the component the reflector just removed, recomputing from
     *  the column when the downdate has drifted too far to trust. */
    private fun downdateNorms(buf: DoubleArray, m: Int, n: Int, col: Int, current: DoubleArray, computed: DoubleArray) {
        for (c in col + 1 until n) {
            if (current[c] == 0.0) continue
            val ratio = abs(buf[col + c * m]) / current[c]
            val remaining = maxOf(0.0, (1.0 - ratio) * (1.0 + ratio))
            val drift = remaining * (current[c] / computed[c]).let { it * it }
            if (drift <= NORM_RECOMPUTE_THRESHOLD) {
                val len = m - col - 1
                val base = col + 1 + c * m
                current[c] = if (len > 0) vectorKernels.nrm2(buf, base, len) else 0.0
                computed[c] = current[c]
            } else {
                current[c] *= sqrt(remaining)
            }
        }
    }

    /** Build the Householder reflector for [col] in place (LAPACK `dlarfg`), returning `tau` and storing
     *  the scaled vector below the diagonal with the new `R` diagonal entry on it. */
    private fun householderColumn(buf: DoubleArray, m: Int, col: Int): Double {
        val base = col + col * m
        val len = m - col
        val norm = vectorKernels.nrm2(buf, base, len)
        if (norm == 0.0) return 0.0
        val alpha = buf[base]
        val beta = if (alpha >= 0.0) -norm else norm
        // Division keeps a subnormal v0 from becoming an infinity.
        val v0 = alpha - beta
        for (i in base + 1 until base + len) buf[i] /= v0
        buf[base] = beta
        return (beta - alpha) / beta
    }

    override fun applyQInto(qr: QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        val m = qr.m
        requireShape(y.size == m) { "applyQ: y length ${y.size} != $m" }
        requireShape(out.size == m) { "applyQ: out length ${out.size} != $m" }
        val x = out
        if (out !== y) y.copyInto(out)
        val k = qr.tau.size
        val buf = qr.qr
        val order = if (transpose) 0 until k else k - 1 downTo 0
        for (col in order) {
            val t = qr.tau[col]
            if (t == 0.0) continue
            val len = m - col - 1
            val vBase = col + 1 + col * m
            val s = x[col] + vectorKernels.dot(buf, vBase, x, col + 1, len)
            val f = t * s
            x[col] -= f
            if (f != 0.0) vectorKernels.axpy(x, col + 1, -f, buf, vBase, len)
        }
        return x
    }

    override fun factor(a: DenseMatrix): LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        return factorCore(a, LuDecomposition(n, DoubleArray(n * n), IntArray(n)))
    }

    override fun factorInto(a: DenseMatrix, out: LuDecomposition): LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        return factorCore(a, out)
    }

    /** Doolittle LU with partial pivoting, writing into [out]'s buffers. */
    private fun factorCore(a: DenseMatrix, out: LuDecomposition): LuDecomposition {
        val n = out.n
        val lu = out.lu
        a.data.copyInto(lu)
        val piv = out.piv
        val kernels = vectorKernels
        for (i in 0 until n) piv[i] = i
        var failedAt = NOT_SINGULAR
        for (k in 0 until n) {
            var p = k
            var max = abs(lu[k + k * n])
            for (i in k + 1 until n) {
                val v = abs(lu[i + k * n])
                if (v > max) {
                    max = v
                    p = i
                }
            }
            if (max == 0.0) {
                if (failedAt == NOT_SINGULAR) failedAt = k
                continue
            }
            if (p != k) {
                for (j in 0 until n) {
                    val t = lu[k + j * n]
                    lu[k + j * n] = lu[p + j * n]
                    lu[p + j * n] = t
                }
                val tp = piv[k]
                piv[k] = piv[p]
                piv[p] = tp
            }
            // Division keeps a subnormal pivot from becoming an infinity.
            val pivot = lu[k + k * n]
            val len = n - k - 1
            val colBase = k + 1 + k * n
            for (i in k + 1 until n) lu[i + k * n] = lu[i + k * n] / pivot
            for (j in k + 1 until n) {
                val ukj = lu[k + j * n]
                if (ukj != 0.0) kernels.axpy(lu, k + 1 + j * n, -ukj, lu, colBase, len)
            }
        }
        out.failedAt = failedAt
        return out
    }

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: LuDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ): DoubleArray {
        requireFactored(lu.failedAt, "solve")
        val n = lu.n
        requireShape(b.size == n) { "solve: b length ${b.size} != $n" }
        requireShape(out.size == n) { "solve: out length ${out.size} != $n" }
        val a = lu.lu
        val piv = lu.piv
        return if (transpose) {
            solveTranspose(n, a, piv, b, out, workspace)
        } else {
            solveNormal(n, a, piv, b, out, workspace)
        }
    }

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    private fun solveNormal(
        n: Int,
        a: DoubleArray,
        piv: IntArray,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray {
        // When out aliases b the permutation may read a slot already written, so it stages the gather.
        if (out === b) {
            val staged = workspace?.take(n) ?: DoubleArray(n)
            for (i in 0 until n) staged[i] = b[piv[i]]
            staged.copyInto(out)
            workspace?.release(staged)
        } else {
            for (i in 0 until n) out[i] = b[piv[i]]
        }
        trsvCore(vectorKernels, a, n, out, lower = true, transpose = false, unitDiag = true)
        trsvCore(vectorKernels, a, n, out, lower = false, transpose = false, unitDiag = false)
        return out
    }

    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    private fun solveTranspose(
        n: Int,
        a: DoubleArray,
        piv: IntArray,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
    ): DoubleArray {
        val y = workspace?.take(n) ?: DoubleArray(n)
        b.copyInto(y)
        trsvCore(vectorKernels, a, n, y, lower = false, transpose = true, unitDiag = false)
        trsvCore(vectorKernels, a, n, y, lower = true, transpose = true, unitDiag = true)
        for (i in 0 until n) out[piv[i]] = y[i] // undo the row permutation
        workspace?.release(y)
        return out
    }
}
