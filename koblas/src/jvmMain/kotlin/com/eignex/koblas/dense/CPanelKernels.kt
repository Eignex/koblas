package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings

/** Exact native panel leaves; performance policy is resolved before entering this implementation. */
internal class CPanelKernels(private val bindings: JvmCKernelBindings) : DensePanelKernels {
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
    ) = bindings.denseDot4(a, aOff, stride, b, bOff, len, out, outOff)

    @Suppress("LongParameterList")
    override fun axpy4(
        y: DoubleArray,
        yOff: Int,
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        c0: Double,
        c1: Double,
        c2: Double,
        c3: Double,
        len: Int,
    ) = bindings.denseAxpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)

    @Suppress("LongParameterList")
    override fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double = bindings.denseDotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        bindings.denseAxpyArithmetic(y, yOff, alpha, x, xOff, len)
}
