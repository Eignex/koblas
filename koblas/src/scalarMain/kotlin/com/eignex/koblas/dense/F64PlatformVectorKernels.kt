package com.eignex.koblas.dense

import com.eignex.koblas.absoluteSum
import com.eignex.koblas.euclideanNorm
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The compile-time kernels for every non-JVM target. Routing to a registered [F64VectorKernels] backend
 * happens above these, in [F64RoutedVectorKernels], so these are pure loops with no dispatch.
 */

internal actual object F64PlatformVectorKernels : F64VectorKernels {
    actual override val name: String get() = BackendNames.SCALAR

    override val isPortable: Boolean get() = true

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        scalarDot(a, aOff, b, bOff, len)

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        scalarAxpy(y, yOff, alpha, x, xOff, len)

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) = scalarScale(v, vOff, alpha, len)

    /** Shared with every other target, since these two have no per-target specialization. */
    actual override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = euclideanNorm(v, vOff, len)

    /** See [nrm2]. */
    actual override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)

    /** See [scalarDot4]: four accumulators over one pass, so the shared operand is read once. */
    @Suppress("LongParameterList") // four column offsets plus the shared operand
    actual override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) = scalarDot4(a, aOff, stride, b, bOff, len, out, outOff)
}
