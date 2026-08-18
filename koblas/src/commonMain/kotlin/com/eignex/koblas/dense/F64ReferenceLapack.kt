@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.BackendNames
import com.eignex.koblas.F64Context
import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import kotlin.math.abs
import kotlin.math.sqrt

/** Bunch-Kaufman pivot threshold `(1 + sqrt(17)) / 8`, the value minimizing element growth (netlib dsytf2). */
private const val BUNCH_KAUFMAN_ALPHA = 0.6403882032022076

/**
 * The portable dense factorizations, the semantic reference a native [F64Lapack] is validated against.
 *
 * @param kernels the vector kernels the inner loops use, or null to follow the [F64Context] default.
 */
internal class F64ReferenceLapack(private val kernels: F64VectorKernels? = null) : F64Lapack {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** These routines' kernels, or the process default when they were given none. */
    override val vectorKernels: F64VectorKernels get() = kernels ?: koblas.vectorKernels

    @Suppress("CyclomaticComplexMethod") // netlib dsytf2's control flow, kept recognizable
    override fun ldl(a: F64DenseMatrix, workspace: Workspace?): F64LdlDecomposition {
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
        return F64LdlDecomposition(n, w, ipiv, failedAt)
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

    override fun solveInto(ldl: F64LdlDecomposition, b: DoubleArray, out: DoubleArray): DoubleArray {
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

    override fun qr(a: F64DenseMatrix, workspace: Workspace?): F64QrDecomposition {
        val m = a.rows
        val n = a.cols
        val k = minOf(m, n)
        val buf = a.data.copyOf()
        val tau = DoubleArray(k)
        for (col in 0 until k) {
            tau[col] = householderColumn(buf, m, col)
            applyReflectorToTrailing(buf, m, n, col, tau[col])
        }
        return F64QrDecomposition(m, n, buf, tau)
    }

    override fun qrPivoted(a: F64DenseMatrix, tolerance: Double, workspace: Workspace?): F64PivotedQrDecomposition {
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
        return F64PivotedQrDecomposition(
            F64QrDecomposition(m, n, buf, tau),
            pivots,
            rankOfPivotedR(buf, m, n, k, tolerance),
        )
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
            if (drift <= F64_NORM_RECOMPUTE_THRESHOLD) {
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

    override fun applyQInto(qr: F64QrDecomposition, y: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
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

    override fun factor(a: F64DenseMatrix): F64LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        val n = a.rows
        return factorCore(a, F64LuDecomposition(n, DoubleArray(n * n), IntArray(n)))
    }

    override fun factorInto(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
        requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
        requireShape(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
        return factorCore(a, out)
    }

    /** Doolittle LU with partial pivoting, writing into [out]'s buffers. */
    private fun factorCore(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
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
        lu: F64LuDecomposition,
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

    /** Solve `A · X = B`, or `Aᵀ · X = B` when [transpose], into [out], which is returned. [out] may be [b],
     *  and a [workspace] lends the transposed direction's `n·nrhs` staging block. */
    @Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
    override fun solveInto(
        lu: F64LuDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix {
        requireFactored(lu.failedAt, "solve")
        val n = lu.n
        val nrhs = b.cols
        requireSolveShapes(n, b, out)
        if (nrhs == 0) return out
        val f = lu.lu
        return if (transpose) {
            val y = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
            b.data.copyInto(y)
            trsmCore(vectorKernels, f, n, y, nrhs, lower = false, transpose = true, unitDiag = false)
            trsmCore(vectorKernels, f, n, y, nrhs, lower = true, transpose = true, unitDiag = true)
            permuteRows(y, out.data, n, nrhs, lu.piv, gather = false)
            workspace?.release(y)
            out
        } else {
            // The gather cannot read B in place once out aliases it, so that case stages.
            if (out === b) {
                val staged = workspace?.take(n * nrhs) ?: DoubleArray(n * nrhs)
                permuteRows(b.data, staged, n, nrhs, lu.piv, gather = true)
                staged.copyInto(out.data)
                workspace?.release(staged)
            } else {
                permuteRows(b.data, out.data, n, nrhs, lu.piv, gather = true)
            }
            trsmCore(vectorKernels, f, n, out.data, nrhs, lower = true, transpose = false, unitDiag = true)
            trsmCore(vectorKernels, f, n, out.data, nrhs, lower = false, transpose = false, unitDiag = false)
            out
        }
    }

    /** Solve `A · X = B` into [out], which is returned. [out] may be [b]. */
    override fun solveInto(
        ldl: F64LdlDecomposition,
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        workspace: Workspace?,
    ): F64DenseMatrix {
        requireFactored(ldl.failedAt, "solve")
        val n = ldl.n
        val nrhs = b.cols
        requireSolveShapes(n, b, out)
        return solveColumnwise(b, out, n, nrhs, workspace) { col, dst -> solveInto(ldl, col, dst) }
    }

    /** [solveLeastSquares] into [out], which is returned. */
    override fun solveLeastSquaresInto(
        qr: F64PivotedQrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
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

    /** Least-squares solve into [out], which has length `n` and is returned. A [workspace] lends the
     *  length-`m` intermediate for `Qᵀb`. */
    override fun solveLeastSquaresInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
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

    /** Minimum-norm solve into [out], which has length `m` (the wide system's column count) and is returned.
     *  A [workspace] lends the length-`n` intermediate. */
    override fun solveMinimumNormInto(
        qr: F64QrDecomposition,
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace?,
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

    override fun rcond(lu: F64LuDecomposition, anorm: Double, workspace: Workspace?): Double {
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

    /** Cholesky factorization `A = L·Lᵀ` of a symmetric positive-definite [a] (`dpotrf` with `uplo = 'L'`).
     *  Reads only the lower triangle of [a], so an upper-only matrix factors to silent nonsense.
     *  @throws com.eignex.koblas.NotPositiveDefinite at the first non-positive pivot unless [policy] allows it.
     */
    override fun cholesky(a: F64DenseMatrix, policy: CholeskyPolicy): F64CholeskyDecomposition {
        requireShape(a.rows == a.cols) { "cholesky requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        val l = F64DenseMatrix(n, n)
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
        return F64CholeskyDecomposition(l)
    }

    /** Invert a general matrix from its LU factorization, returning `A⁻¹` given `P·A = L·U` (LAPACK `dgetri`).
     *  Prefer [solve] to apply `A⁻¹`, which costs less and is more accurate.
     *  @throws com.eignex.koblas.SingularMatrix if [lu] is singular; the position is [F64LuDecomposition.failedAt].
     */
    override fun invert(lu: F64LuDecomposition, workspace: Workspace?): F64DenseMatrix {
        if (lu.singular) {
            throw SingularMatrix(
                lu.failedAt,
                "invert: factorization is singular at pivot ${lu.failedAt}, so the inverse does not exist",
            )
        }
        val n = lu.n
        val inv = F64DenseMatrix.diagonal(n)
        return solveInto(lu, inv, inv, transpose = false, workspace = workspace)
    }

    /** Invert a triangular matrix into a fresh result (LAPACK `dtrtri`), returning `T⁻¹` for the [lower] or
     *  upper triangle of the square [a], taking the diagonal as 1 when [unitDiag].
     *  @throws com.eignex.koblas.SingularMatrix naming the first zero diagonal position.
     */
    override fun trtri(a: F64DenseMatrix, lower: Boolean, unitDiag: Boolean): F64DenseMatrix {
        requireShape(a.rows == a.cols) { "trtri requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        if (!unitDiag) {
            for (i in 0 until n) {
                if (a[i, i] == 0.0) {
                    throw SingularMatrix(i, "trtri: triangle is singular, diagonal entry $i is zero")
                }
            }
        }
        val inv = F64DenseMatrix(n, n)
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
    override fun invert(chol: F64CholeskyDecomposition, workspace: Workspace?): F64DenseMatrix {
        val n = chol.n
        val ld = chol.l.data
        val inv = F64DenseMatrix(n, n)
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

    /** The estimator body, over caller-supplied vectors so [rcond] can pool them. */
    @Suppress("LongParameterList") // four scratch vectors the caller owns
    private fun hagerEstimate(
        lu: F64LuDecomposition,
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
}
