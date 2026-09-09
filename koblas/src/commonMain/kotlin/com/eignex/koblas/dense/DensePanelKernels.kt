package com.eignex.koblas.dense

/** Arithmetic over matrix columns and panels, including updates that must evaluate a zero multiplier. */
public interface DensePanelKernels {
    /** Writes four dot products against one shared right operand into [out]. */
    @Suppress("LongParameterList")
    public fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    )

    /** Adds four scaled, equally spaced matrix columns into [y], evaluating zero coefficients. */
    @Suppress("LongParameterList")
    public fun axpy4(
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
    )

    /** Returns one dot while adding `alpha * a` into [y] in the same pass, evaluating zero [alpha]. */
    @Suppress("LongParameterList")
    public fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double

    /** Always evaluates `alpha * x`, including when [alpha] is zero. */
    @Suppress("LongParameterList")
    public fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)
}
