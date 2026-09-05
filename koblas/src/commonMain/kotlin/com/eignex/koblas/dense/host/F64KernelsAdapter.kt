package com.eignex.koblas.dense.host

import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.portableRotmg
import com.eignex.koblas.toBlasParameters

/**
 * The level-1 routines a host CBLAS provides, over whichever [CblasKernelCalls] the platform supplies. Both
 * host bindings are this class plus their own FFI mechanism, as [F64BlasAdapter] is one level up.
 *
 * koblas applies the selected kernel backend, so these methods see every vector run, but the [F64Kernels]
 * contract makes a length of zero legal everywhere and an override of the threshold routes those here. Each
 * routine answers one itself: an empty run sits at the end of its array, where there is no element to take an
 * address of.
 */
public abstract class F64KernelsAdapter internal constructor(private val f: CblasKernelCalls) : F64Kernels {

    /** Above the reference (0) and the native dlopen backend (90). */
    override val priority: Int get() = HOST_BACKEND_PRIORITY

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (len == 0) 0.0 else f.ddot(len, a, aOff, 1, b, bOff, 1)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len != 0) f.daxpy(len, alpha, x, xOff, 1, y, yOff, 1)
    }

    /** OpenBLAS has no plain sum, only `dasum` over absolute values, so this runs the compiled-in kernel. */
    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double = F64PlatformKernels.sum(v, vOff, len)

    /**
     * OpenBLAS has no fused sum of squared differences, and the only way to build one from the routines it
     * does have is `a.a - 2a.b + b.b`, whose cancellation ruins exactly the small values this is for. So
     * this one routine runs the compiled-in kernel instead of the host library.
     */
    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        F64PlatformKernels.ssqd(a, aOff, b, bOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (len != 0) f.dswap(len, a, aOff, 1, b, bOff, 1)
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (len != 0) f.dscal(len, alpha, v, vOff, 1)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = if (len == 0) 0.0 else f.dnrm2(len, v, vOff, 1)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = if (len == 0) 0.0 else f.dasum(len, v, vOff, 1)

    // OpenBLAS's rescaling state differs from Netlib's, so the public state contract stays portable.
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens =
        portableRotmg(d1, d2, x1, y1)

    @Suppress("LongParameterList") // the BLAS drotm signature
    override fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: F64ModifiedGivens,
    ) {
        if (len == 0 || transformation.flag == -2.0) return
        f.drotm(
            len,
            x,
            blasOffset(xOff, xStride, len),
            xStride,
            y,
            blasOffset(yOff, yStride, len),
            yStride,
            transformation.toBlasParameters(),
        )
    }

    @Suppress("LongParameterList") // the BLAS drot signature
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        if (len == 0) return
        f.drot(len, x, xOff, 1, y, yOff, 1, c, s)
    }
}
