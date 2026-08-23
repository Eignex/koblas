package com.eignex.koblas.hostblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.dense.host.cblas.OpenBlasConfig
import com.eignex.koblas.dense.host.jvm.HostBlasCalls
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.backend.openBlasDispatchThresholds

/** The level-1 CBLAS routines from the OpenBLAS instance shared with [HostBlas]. */
public class HostVectorKernels internal constructor(
    private val calls: HostBlasCalls,
    /** Policy for this level-1 backend instance. */
    public val config: OpenBlasConfig,
) : F64VectorKernels {
    public constructor(config: OpenBlasConfig = OpenBlasConfig()) : this(HostBlasCalls(config), config)

    override val name: String get() = BackendNames.OPENBLAS
    override val priority: Int get() = HOST_BACKEND_PRIORITY
    override val minDispatchLength: Int get() = openBlasDispatchThresholds(config).level1
    override val isAvailable: Boolean get() = calls.available

    private fun segment(values: DoubleArray, offset: Int) =
        java.lang.foreign.MemorySegment.ofArray(values).asSlice(offset.toLong() * Double.SIZE_BYTES)

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (len == 0) 0.0 else calls.ddot.invokeWithArguments(len, segment(a, aOff), 1, segment(b, bOff), 1) as Double

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len != 0) calls.daxpy.invokeWithArguments(len, alpha, segment(x, xOff), 1, segment(y, yOff), 1)
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (len != 0) calls.dscal.invokeWithArguments(len, alpha, segment(v, vOff), 1)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else calls.dnrm2.invokeWithArguments(len, segment(v, vOff), 1) as Double

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len == 0) 0.0 else calls.dasum.invokeWithArguments(len, segment(v, vOff), 1) as Double

    override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        repeat(4) { i -> out[outOff + i] = dot(a, aOff + i * stride, b, bOff, len) }
    }
}
