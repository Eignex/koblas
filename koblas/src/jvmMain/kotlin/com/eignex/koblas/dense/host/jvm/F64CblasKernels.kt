package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.dense.host.blasOffset
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.portableRotmg
import com.eignex.koblas.toBlasParameters

/** The level-1 CBLAS routines from the OpenBLAS instance shared with [F64Cblas]. */
public class F64CblasKernels internal constructor(
    private val calls: HostBlasCalls,
    /** Policy for this level-1 backend instance. */
    public val config: HostBlasConfig,
) : F64Kernels {
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(HostBlasCalls(config), config)

    override val name: String get() = BackendNames.OPENBLAS
    override val priority: Int get() = HOST_BACKEND_PRIORITY
    override val isAvailable: Boolean get() = calls.available

    private fun segment(values: DoubleArray, offset: Int) =
        java.lang.foreign.MemorySegment.ofArray(values).asSlice(offset.toLong() * Double.SIZE_BYTES)

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (len == 0) 0.0 else calls.ddot.invokeExact(len, segment(a, aOff), 1, segment(b, bOff), 1) as Double

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len != 0) calls.daxpy.invokeExact(len, alpha, segment(x, xOff), 1, segment(y, yOff), 1) as Unit
    }

    /**
     * OpenBLAS has no fused sum of squared differences, and the only way to build one from the routines it
     * does have is `a.a - 2a.b + b.b`, whose cancellation ruins exactly the small values this is for. So
     * this one routine runs the compiled-in kernel instead of the host library.
     */
    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        F64PlatformKernels.ssqd(a, aOff, b, bOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (len != 0) calls.dswap.invokeExact(len, segment(a, aOff), 1, segment(b, bOff), 1) as Unit
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (len != 0) calls.dscal.invokeExact(len, alpha, segment(v, vOff), 1) as Unit
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else calls.dnrm2.invokeExact(len, segment(v, vOff), 1) as Double

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else calls.dasum.invokeExact(len, segment(v, vOff), 1) as Double

    // OpenBLAS's rescaling state differs from Netlib's, so the public state contract stays portable.
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens =
        portableRotmg(d1, d2, x1, y1)

    @Suppress("LongParameterList")
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
        val parameters = transformation.toBlasParameters()
        calls.drotm.invokeExact(
            len,
            segment(x, blasOffset(xOff, xStride, len)),
            xStride,
            segment(y, blasOffset(yOff, yStride, len)),
            yStride,
            segment(parameters, 0),
        ) as Unit
    }
}
