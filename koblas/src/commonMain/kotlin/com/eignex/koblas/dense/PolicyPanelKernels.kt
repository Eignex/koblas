package com.eignex.koblas.dense

/** Shared panel dispatch; arithmetic stays in the independently callable implementations. */
internal class PolicyPanelKernels(
    private val runtime: DensePanelKernels,
    private val native: DensePanelKernels,
    private val dispatch: DenseDispatch,
) : DensePanelKernels {
    private fun selected(operation: DenseOperation, length: Int): DensePanelKernels =
        if (dispatch.usesNative(operation, length)) native else runtime

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
    ) = selected(DenseOperation.Dot4, len).dot4(a, aOff, stride, b, bOff, len, out, outOff)

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
    ) = selected(DenseOperation.Axpy4, len).axpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)

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
    ): Double = selected(DenseOperation.DotAxpy, len).dotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        selected(DenseOperation.AxpyArithmetic, len).axpyArithmetic(y, yOff, alpha, x, xOff, len)
}
