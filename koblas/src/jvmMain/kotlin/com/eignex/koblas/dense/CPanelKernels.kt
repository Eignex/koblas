package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.internal.numeric.scalarAxpy4
import com.eignex.koblas.internal.numeric.scalarAxpyArithmetic
import com.eignex.koblas.internal.numeric.scalarDot4
import com.eignex.koblas.internal.numeric.scalarDotAxpy

/** Bundled-C matrix-panel arithmetic with measured portable JVM fallbacks. */
internal class CPanelKernels(private val bindings: JvmCKernelBindings, private val exact: Boolean) : DensePanelKernels {
    private val dot4CCrossover = DenseTuning.jvmCDot4Crossover
    private val axpy4CCrossover = DenseTuning.jvmCAxpy4Crossover
    private val dotAxpyCCrossover = DenseTuning.jvmCDotAxpyCrossover

    override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) = if (!exact && len < dot4CCrossover) {
        scalarDot4(a, aOff, stride, b, bOff, len, out, outOff)
    } else {
        bindings.denseDot4(a, aOff, stride, b, bOff, len, out, outOff)
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
    ) = if (!exact && len < axpy4CCrossover) {
        scalarAxpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
    } else {
        bindings.denseAxpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
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
    ): Double = if (!exact && len < dotAxpyCCrossover) {
        scalarDotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)
    } else {
        bindings.denseDotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)
    }

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        if (exact) {
            bindings.denseAxpyArithmetic(y, yOff, alpha, x, xOff, len)
        } else {
            scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
        }
}
