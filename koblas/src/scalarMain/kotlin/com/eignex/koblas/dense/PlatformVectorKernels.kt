package com.eignex.koblas.dense

import com.eignex.koblas.absoluteSum
import com.eignex.koblas.euclideanNorm

/**
 * The compile-time kernels for every non-JVM target. Routing to a registered [VectorKernels] backend
 * happens above these, in [RoutedVectorKernels], so these are pure loops with no dispatch.
 */

internal actual object PlatformVectorKernels : VectorKernels {
    actual override val name: String get() = "scalar"

    actual override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        var s = 0.0
        for (i in 0 until len) s += a[aOff + i] * b[bOff + i]
        return s
    }

    actual override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
    }

    actual override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (alpha == 1.0) return
        for (i in 0 until len) v[vOff + i] *= alpha
    }

    /** Shared with every other target, since these two have no per-target specialization. */
    actual override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = euclideanNorm(v, vOff, len)

    /** See [nrm2]. */
    actual override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)

    /** Four accumulators over one pass, so the shared b element is read once for all four columns. */
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
    ) {
        val o1 = aOff + stride
        val o2 = aOff + 2 * stride
        val o3 = aOff + 3 * stride
        var r0 = 0.0
        var r1 = 0.0
        var r2 = 0.0
        var r3 = 0.0
        for (i in 0 until len) {
            val bi = b[bOff + i]
            r0 += a[aOff + i] * bi
            r1 += a[o1 + i] * bi
            r2 += a[o2 + i] * bi
            r3 += a[o3 + i] * bi
        }
        out[outOff] = r0
        out[outOff + 1] = r1
        out[outOff + 2] = r2
        out[outOff + 3] = r3
    }
}
