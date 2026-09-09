package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.internal.numeric.scalarAxpy4
import com.eignex.koblas.internal.numeric.scalarAxpyArithmetic
import com.eignex.koblas.internal.numeric.scalarDot4
import com.eignex.koblas.internal.numeric.scalarDotAxpy

/** Bundled-C matrix-panel arithmetic with measured portable JVM fallbacks. */
internal object CPanelKernels : DensePanelKernels {
    private val DOT4_C_CROSSOVER = DenseTuning.jvmCDot4Crossover
    private val AXPY4_C_CROSSOVER = DenseTuning.jvmCAxpy4Crossover
    private val DOT_AXPY_C_CROSSOVER = DenseTuning.jvmCDotAxpyCrossover

    override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) = if (len < DOT4_C_CROSSOVER) {
        scalarDot4(a, aOff, stride, b, bOff, len, out, outOff)
    } else {
        JvmCKernelBindings.denseDot4(a, aOff, stride, b, bOff, len, out, outOff)
    }

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
    ) = if (len < AXPY4_C_CROSSOVER) {
        scalarAxpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
    } else {
        JvmCKernelBindings.denseAxpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
    }

    override fun dotAxpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        len: Int,
    ): Double = if (len < DOT_AXPY_C_CROSSOVER) {
        scalarDotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)
    } else {
        JvmCKernelBindings.denseDotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)
    }

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
}
