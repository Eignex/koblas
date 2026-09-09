package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.scalarAxpy4
import com.eignex.koblas.internal.numeric.scalarAxpyArithmetic
import com.eignex.koblas.internal.numeric.scalarDot4
import com.eignex.koblas.internal.numeric.scalarDotAxpy

/** JVM Vector API matrix-panel arithmetic with scalar short-run fallbacks. */
internal object SimdPanelKernels : DensePanelKernels {
    private val lanes: Int = if (simdAvailable) SimdOps.lanes() else 0

    private fun vectorizes(len: Int): Boolean = simdAvailable && len >= lanes

    override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        if (vectorizes(len)) {
            SimdOps.dot4(a, aOff, stride, b, bOff, len, out, outOff)
        } else {
            scalarDot4(a, aOff, stride, b, bOff, len, out, outOff)
        }
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
    ) {
        if (vectorizes(len)) {
            SimdOps.axpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
        } else {
            scalarAxpy4(y, yOff, a, aOff, stride, c0, c1, c2, c3, len)
        }
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
    ): Double = if (vectorizes(len)) {
        SimdOps.dotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)
    } else {
        scalarDotAxpy(y, yOff, alpha, a, aOff, x, xOff, len)
    }

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (vectorizes(len)) {
            SimdOps.axpyArithmetic(y, yOff, alpha, x, xOff, len)
        } else {
            scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
        }
    }
}
