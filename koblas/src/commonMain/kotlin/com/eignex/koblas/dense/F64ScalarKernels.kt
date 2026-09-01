package com.eignex.koblas.dense

import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.numeric.*

/** Pure Kotlin scalar kernels retained as the portable fallback and semantic reference for compiled leaves. */
internal object F64ScalarKernels : F64Kernels, F64ArithmeticKernels {
    override val name: String get() = BackendNames.SCALAR

    override val isPortable: Boolean get() = true

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        scalarDot(a, aOff, b, bOff, len)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        scalarAxpy(y, yOff, alpha, x, xOff, len)

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) = scalarScale(v, vOff, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = euclideanNorm(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)

    @Suppress("LongParameterList")
    override fun dot4(
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
