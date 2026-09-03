@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.dense

import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.portableRotmg
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/** The C level-1 kernels compiled into each Kotlin/Native host artifact. */
internal actual object F64PlatformKernels : F64Kernels, F64ArithmeticKernels {
    actual override val name: String get() = BackendNames.C

    override val isPortable: Boolean get() = true

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (len == 0) {
        0.0
    } else {
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_dot(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len == 0 || alpha == 0.0) return
        y.usePinned { yp ->
            x.usePinned { xp -> koblas_dense_axpy(yp.addressOf(0), yOff, alpha, xp.addressOf(0), xOff, len) }
        }
    }

    actual override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len == 0) return
        y.usePinned { yp ->
            x.usePinned { xp -> koblas_dense_axpy_arithmetic(yp.addressOf(0), yOff, alpha, xp.addressOf(0), xOff, len) }
        }
    }

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (len == 0 || alpha == 1.0) return
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
