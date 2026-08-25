@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.internal.numeric.F64_MIN_NORMAL
import kotlin.math.abs

/*
 * The portable Bunch-Kaufman factorization and the solve over its factors, netlib dsytf2 and dsytrs. This is
 * the semantic definition a native LDL is validated against; [F64ReferenceDecompositions] is the
 * F64Decompositions surface over it.
 *
 * One deliberate departure: the 1x1 elimination takes dsytf2_rook's guarded scaling rather than plain
 * dsytf2's unguarded reciprocal, so a subnormal pivot factors here where a host dsytrf answers with
 * infinities. See [eliminateOneByOne].
 */

/** Bunch-Kaufman pivot threshold `(1 + sqrt(17)) / 8`, the value minimizing element growth (netlib dsytf2). */
private const val BUNCH_KAUFMAN_ALPHA = 0.6403882032022076

@Suppress("CyclomaticComplexMethod") // netlib dsytf2's control flow, kept recognizable
internal fun referenceLdl(kernels: F64Kernels, a: F64DenseMatrix, workspace: Workspace?): F64LdlDecomposition {
    requireShape(a.rows == a.cols) { "ldl: matrix must be square, got ${a.rows}x${a.cols}" }
    val n = a.rows
    val w = a.data.copyOf()
    val ipiv = IntArray(n)
    var failedAt = NOT_SINGULAR
    workspace.borrow(n) { colK ->
        workspace.borrow(n) { colK1 ->
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
                if (maxOf(absakk, colmax) == 0.0 || absakk.isNaN()) {
                    // The first zero pivot is the one reported, matching dsytf2's info, which reports a NaN
                    // diagonal the same way rather than eliminating on it.
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
                    // Grouped as dsytf2 groups it. The algebraically equal `absakk * rowmax >= alpha * colmax
                    // * colmax` leaves the exponent range on a matrix this one holds: rowmax >= colmax here, so
                    // whichever side overflows first the other follows, and `Inf >= Inf` selects the 1x1 pivot
                    // this test exists to reject. rowmax > 0 because the scan below covers the entry colmax was
                    // attained at.
                    if (absakk >= BUNCH_KAUFMAN_ALPHA * colmax * (colmax / rowmax)) {
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
                    eliminateOneByOne(kernels, w, n, k)
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
        }
    }
    return F64LdlDecomposition(n, w, ipiv, failedAt)
}

/**
 * Eliminate a 1x1 pivot and update the trailing triangle, `dsytf2`'s `dscal` and `dsyr`.
 *
 * A pivot below the smallest normal has no reciprocal: `1 / 1e-320` is an infinity, and scaling the column
 * by it turns a factor that dividing resolves exactly into infinities and NaNs. So the small case divides
 * the column and scales the update by the pivot itself, which is the same arithmetic out of the reciprocal's
 * exponent range, and is the branch dsytf2 takes below its own `sfmin`.
 */
private fun eliminateOneByOne(kernels: F64Kernels, w: DoubleArray, n: Int, k: Int) {
    val pivot = w[k + k * n]
    // A pivot below the smallest normal has no reciprocal: `1 / 1e-320` is an infinity, and scaling the
    // column by it turns multipliers that dividing resolves exactly into infinities. So the small case
    // divides the column first and scales the update by the pivot itself, which is the same arithmetic out
    // of the reciprocal's exponent range. That is dsytf2_rook's branch below its own sfmin; plain dsytf2
    // takes the reciprocal whatever the pivot. It bounds the reciprocal, not the result: an entry far above
    // the pivot still overflows the division, which the pivot search allows on its rowmax branch.
    val scalable = abs(pivot) >= F64_MIN_NORMAL
    if (!scalable) {
        for (i in k + 1 until n) w[i + k * n] = w[i + k * n] / pivot
    }
    // Against the divided column the update takes the pivot where the reciprocal form takes its inverse.
    val coefficient = if (scalable) 1.0 / pivot else pivot
    for (j in k + 1 until n) {
        val f = -coefficient * w[j + k * n]
        if (f != 0.0) kernels.axpy(w, j + j * n, f, w, j + k * n, n - j)
    }
    if (scalable) kernels.scale(w, k + 1 + k * n, coefficient, n - k - 1)
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

internal fun referenceLdlSolveInto(
    kernels: F64Kernels,
    ldl: F64LdlDecomposition,
    b: DoubleArray,
    out: DoubleArray,
): DoubleArray {
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
            if (xk != 0.0) kernels.axpy(x, k + 1, -xk, w, k + 1 + k * n, n - k - 1)
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
            if (xk != 0.0) kernels.axpy(x, k + 2, -xk, w, k + 2 + k * n, len)
            if (xk1 != 0.0) kernels.axpy(x, k + 2, -xk1, w, k + 2 + (k + 1) * n, len)
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
            x[k] -= kernels.dot(w, k + 1 + k * n, x, k + 1, n - k - 1)
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
            val s0 = x[k0] - kernels.dot(w, k + 1 + k0 * n, x, k + 1, len)
            val s1 = x[k] - kernels.dot(w, k + 1 + k * n, x, k + 1, len)
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
