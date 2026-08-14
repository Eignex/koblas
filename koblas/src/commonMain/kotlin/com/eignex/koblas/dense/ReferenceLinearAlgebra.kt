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

/** Bunch–Kaufman pivot threshold `(1 + √17) / 8`, the value minimizing element growth (netlib dsytf2). */
private const val BUNCH_KAUFMAN_ALPHA = 0.6403882032022076

/**
 * Portable pure-Kotlin backend, correct on every target with no native dependency, and the semantic
 * reference a native backend is validated against.
 *
 * @param kernels the vector kernels the inner loops use, or null to follow the [KoblasContext] default.
 */
public class ReferenceBackend(private val kernels: VectorKernels? = null) : LinearAlgebra {
    override val name: String get() = "reference"

    /** This backend's kernels, or the process default when it was given none. */
    override val vectorKernels: VectorKernels get() = kernels ?: koblas.vectorKernels

    /** Whether these routines follow the kernels around them rather than ones of their own. */
    internal val followsContext: Boolean get() = kernels == null

    override fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        if (beta == 0.0) {
            y.fill(0.0)
        } else if (beta != 1.0) {
            vectorKernels.scale(y, 0, beta, y.size)
        }
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
        if (beta == 0.0) {
            cd.fill(0.0)
        } else if (beta != 1.0) {
            vectorKernels.scale(cd, 0, beta, cd.size)
        }
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
        if (beta == 0.0) {
            y.fill(0.0)
        } else if (beta != 1.0) {
            vectorKernels.scale(y, 0, beta, n)
        }
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
        if (beta == 0.0) {
            cd.fill(0.0)
        } else if (beta != 1.0) {
            vectorKernels.scale(cd, 0, beta, cd.size)
        }
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

    @Suppress("CyclomaticComplexMethod", "LongMethod") // netlib dsytf2's control flow, kept recognizable
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
            if (kp != kk) {
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
        // Divided rather than scaled by a reciprocal, so a subnormal v0 cannot overflow into an infinity.
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
            // Divided rather than reciprocal-scaled, so a subnormal pivot cannot turn into an infinity.
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

/** The shared portable backend, the fallback every seam resolves to when nothing else is registered. */
public val ReferenceLinearAlgebra: ReferenceBackend = ReferenceBackend()
