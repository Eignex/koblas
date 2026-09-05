@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.dense

import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.numeric.scalarAxpy
import com.eignex.koblas.internal.numeric.scalarAxpyArithmetic
import com.eignex.koblas.internal.numeric.scalarDot
import com.eignex.koblas.internal.numeric.scalarScale
import com.eignex.koblas.portableRotmg
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/**
 * The shortest run for which crossing into the C kernels pays on Kotlin/Native.
 *
 * Every call here pins its arrays and crosses a foreign boundary, and below `KOBLAS_UNROLL_MIN` the C
 * kernel runs the same plain loop the scalar helper would, so a short run buys the overhead and nothing
 * else. Measured on `Level1Benchmark` for linuxX64, C against scalar: at len 2 dot is 40.7 ns against 19.2
 * and at 8 it is 39.8 against 22.4, both losses; at 32 they are level (43.4 against 40.2); at 48 C is ahead
 * on all three measured routines (dot 47.7 against 51.3, axpy 39.2 against 49.2, nrm2 34.5 against 41.8)
 * and pulls away from there, reaching 2x by 128. This takes the conservative end of that range.
 *
 * The JVM half already gates this way, at its own measured 128 in `F64CKernels`.
 */
private const val C_HOST_MIN_LENGTH = 48

/** The C level-1 kernels compiled into each Kotlin/Native host artifact. */
internal actual object F64PlatformKernels : F64Kernels, F64ArithmeticKernels {
    actual override val name: String get() = BackendNames.C

    override val isPortable: Boolean get() = true

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (
        len < C_HOST_MIN_LENGTH
    ) {
        scalarDot(a, aOff, b, bOff, len)
    } else {
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_dot(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        if (len < C_HOST_MIN_LENGTH) return scalarAxpy(y, yOff, alpha, x, xOff, len)
        y.usePinned { yp ->
            x.usePinned { xp -> koblas_dense_axpy(yp.addressOf(0), yOff, alpha, xp.addressOf(0), xOff, len) }
        }
    }

    actual override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len < C_HOST_MIN_LENGTH) return scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
        y.usePinned { yp ->
            x.usePinned { xp -> koblas_dense_axpy_arithmetic(yp.addressOf(0), yOff, alpha, xp.addressOf(0), xOff, len) }
        }
    }

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (alpha == 1.0) return
        if (len < C_HOST_MIN_LENGTH) return scalarScale(v, vOff, alpha, len)
        v.usePinned { vp -> koblas_dense_scale(vp.addressOf(0), vOff, alpha, len) }
    }

    actual override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> koblas_dense_nrm2(vp.addressOf(0), vOff, len) }

    actual override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> koblas_dense_asum(vp.addressOf(0), vOff, len) }

    actual override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens =
        portableRotmg(d1, d2, x1, y1)

    @Suppress("LongParameterList")
    actual override fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: F64ModifiedGivens,
    ) {
        if (transformation.flag == -2.0 || len == 0) return
        x.usePinned { xp ->
            y.usePinned { yp ->
                koblas_dense_rotm(
                    xp.addressOf(0),
                    xOff,
                    xStride,
                    yp.addressOf(0),
                    yOff,
                    yStride,
                    len,
                    transformation.h11,
                    transformation.h12,
                    transformation.h21,
                    transformation.h22,
                )
            }
        }
    }

    // A plane rotation is the modified Givens transformation (c, s, -s, c), so it goes to the same kernel.
    @Suppress("LongParameterList")
    actual override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        if (len == 0) return
        x.usePinned { xp ->
            y.usePinned { yp ->
                koblas_dense_rotm(xp.addressOf(0), xOff, 1, yp.addressOf(0), yOff, 1, len, c, s, -s, c)
            }
        }
    }

    actual override fun sum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { p -> koblas_dense_sum(p.addressOf(0), vOff, len) }

    actual override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (len == 0) {
        0.0
    } else {
        a.usePinned { ap ->
            b.usePinned { bp ->
                koblas_dense_ssqd(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len)
            }
        }
    }

    actual override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (len == 0) return
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_swap(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }

    @Suppress("LongParameterList")
    actual override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        if (len == 0) {
            for (r in 0 until 4) out[outOff + r] = 0.0
            return
        }
        a.usePinned { ap ->
            b.usePinned { bp ->
                out.usePinned { op ->
                    koblas_dense_dot4(
                        ap.addressOf(0),
                        aOff,
                        stride,
                        bp.addressOf(0),
                        bOff,
                        len,
                        op.addressOf(0),
                        outOff,
                    )
                }
            }
        }
    }
}
