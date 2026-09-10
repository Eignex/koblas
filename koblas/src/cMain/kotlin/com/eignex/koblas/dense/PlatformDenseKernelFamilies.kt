@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
@file:Suppress("MatchingDeclarationName") // Native kernels and their selected family share one source-set boundary

package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.numeric.scalarAxpy
import com.eignex.koblas.internal.numeric.scalarDot
import com.eignex.koblas.internal.numeric.scalarScale
import com.eignex.koblas.portableRotmg
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned

/** Shortest run for which crossing into the C vector kernels pays on Kotlin/Native. */
internal val C_HOST_MIN_LENGTH = DenseTuning.nativeCMinLength

internal actual val platformDenseKernelFamilies: DenseKernelFamilies = DenseKernelFamilies(
    NativeCKernels,
    NativeCPanelKernels,
    NativeCPackedKernels,
)

/** The C vector kernels compiled into each Kotlin/Native host artifact. */
internal object NativeCKernels : DenseVectorKernels {
    override val name: String get() = "c"

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (
        len < C_HOST_MIN_LENGTH
    ) {
        scalarDot(a, aOff, b, bOff, len)
    } else {
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_dot(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        if (len < C_HOST_MIN_LENGTH) return scalarAxpy(y, yOff, alpha, x, xOff, len)
        y.usePinned { yp ->
            x.usePinned { xp -> koblas_dense_axpy(yp.addressOf(0), yOff, alpha, xp.addressOf(0), xOff, len) }
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (alpha == 1.0) return
        if (len < C_HOST_MIN_LENGTH) return scalarScale(v, vOff, alpha, len)
        v.usePinned { vp -> koblas_dense_scale(vp.addressOf(0), vOff, alpha, len) }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> koblas_dense_nrm2(vp.addressOf(0), vOff, len) }

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { vp -> koblas_dense_asum(vp.addressOf(0), vOff, len) }

    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens = portableRotmg(d1, d2, x1, y1)

    @Suppress("LongParameterList")
    override fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: ModifiedGivens,
    ) {
        if (transformation.flag == -2.0 || len == 0) return
        x.usePinned { xp ->
            y.usePinned { yp ->
                koblas_dense_rotm(
                    xp.addressOf(0), xOff, xStride, yp.addressOf(0), yOff, yStride, len,
                    transformation.h11, transformation.h12, transformation.h21, transformation.h22,
                )
            }
        }
    }

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        if (len == 0) return
        x.usePinned { xp ->
            y.usePinned { yp ->
                koblas_dense_rotm(xp.addressOf(0), xOff, 1, yp.addressOf(0), yOff, 1, len, c, s, -s, c)
            }
        }
    }

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else v.usePinned { p -> koblas_dense_sum(p.addressOf(0), vOff, len) }

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (len == 0) {
        0.0
    } else {
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_ssqd(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (len == 0) return
        a.usePinned { ap ->
            b.usePinned { bp -> koblas_dense_swap(ap.addressOf(0), aOff, bp.addressOf(0), bOff, len) }
        }
    }
}
