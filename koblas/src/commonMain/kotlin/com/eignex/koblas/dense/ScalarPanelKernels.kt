package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.scalarAxpy4
import com.eignex.koblas.internal.numeric.scalarAxpyArithmetic
import com.eignex.koblas.internal.numeric.scalarDot4
import com.eignex.koblas.internal.numeric.scalarDotAxpy

/** Pure Kotlin matrix-panel arithmetic and the semantic reference for compiled panel leaves. */
internal object ScalarPanelKernels : DensePanelKernels {
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
    ) = scalarAxpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)

    override fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double = scalarDotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
}
