package com.eignex.koblas.internal.dense

import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.internal.numeric.scalarAxpy
import com.eignex.koblas.internal.numeric.scalarDot
import com.eignex.koblas.internal.numeric.scalarIamax
import com.eignex.koblas.internal.numeric.scalarRot
import com.eignex.koblas.internal.numeric.scalarScale
import com.eignex.koblas.internal.numeric.scalarSum
import com.eignex.koblas.internal.numeric.scalarSwap

/** `v = beta * v` over the [len] entries from [off], honoring the `beta == 0` overwrite convention. */
internal fun applyBeta(k: DenseVectorKernels, v: DoubleArray, off: Int, len: Int, beta: Double, stride: Int = 1) {
    when {
        beta == 0.0 -> if (stride == 1) {
            v.fill(
                0.0,
                off,
                off + len,
            )
        } else {
            for (i in 0 until len) v[off + i * stride] = 0.0
        }

        beta != 1.0 -> k.scale(v, off, beta, len, stride)
    }
}

/**
 * The `scale` component a destination multiplier of [elements] entries reaches, as a route names it.
 *
 * One answer for the dense and sparse schedules, which scale a destination the same way: the leaf the
 * backend resolves for that length, or the backend's own name where it decides on the values instead.
 */
internal fun scaleComponent(vectors: DenseVectorKernels, elements: Int): List<String> =
    listOf("${vectors.implementationFor(DenseOperation.Scale, elements) ?: vectors.name}/scale")

/** Pure Kotlin scalar kernels retained as the portable fallback and semantic reference for compiled leaves. */
internal object ScalarVectorKernels : DenseVectorKernels {
    override val name: String get() = "scalar"

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = scalarDot(a, aOff, aStride, b, bOff, bStride, len)

    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) = scalarAxpy(y, yOff, yStride, alpha, x, xOff, xStride, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) =
        scalarScale(v, vOff, vStride, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = euclideanNorm(v, vOff, vStride, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = absoluteSum(v, vOff, vStride, len)

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int = scalarIamax(v, vOff, vStride, len)

    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = scalarSum(v, vOff, vStride, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        scalarSwap(a, aOff, aStride, b, bOff, bStride, len)

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        scalarRot(x, xOff, y, yOff, len, c, s)
}
