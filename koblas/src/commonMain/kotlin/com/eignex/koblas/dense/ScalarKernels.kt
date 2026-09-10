package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot
import com.eignex.koblas.portableRotm
import com.eignex.koblas.portableRotmg

/** Pure Kotlin scalar kernels retained as the portable fallback and semantic reference for compiled leaves. */
internal object ScalarKernels : DenseVectorKernels {
    override val name: String get() = "scalar"

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        scalarDot(a, aOff, b, bOff, len)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        scalarAxpy(y, yOff, alpha, x, xOff, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) = scalarScale(v, vOff, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = euclideanNorm(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double = scalarSum(v, vOff, len)

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        scalarSsqd(a, aOff, b, bOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        scalarSwap(a, aOff, b, bOff, len)

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
    ) = portableRotm(x, xOff, xStride, y, yOff, yStride, len, transformation)

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        portableRot(x, xOff, y, yOff, len, c, s)
}
