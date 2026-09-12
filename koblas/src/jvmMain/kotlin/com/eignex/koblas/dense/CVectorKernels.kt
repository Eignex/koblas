package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.portableRotmg

/** Exact native vector leaves; performance policy is resolved before entering this implementation. */
internal class CVectorKernels(private val bindings: JvmCKernelBindings) : DenseVectorKernels {
    override val name: String get() = "c-${bindings.variant.name.lowercase()}-raw"

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        if (len == 0) return 0.0
        return bindings.denseDot(a, aOff, b, bOff, len)
    }

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (len == 0 || alpha == 0.0) return
        bindings.denseAxpy(y, yOff, alpha, x, xOff, len)
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (len == 0 || alpha == 1.0) return
        bindings.denseScale(v, vOff, alpha, len)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double {
        if (len == 0) return 0.0
        return bindings.denseNrm2(v, vOff, len)
    }

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double {
        if (len == 0) return 0.0
        return bindings.denseAsum(v, vOff, len)
    }

    override fun iamax(v: DoubleArray, vOff: Int, len: Int): Int {
        if (len == 0) return -1
        return bindings.denseIamax(v, vOff, len)
    }

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
    ) {
        if (transformation.flag == -2.0 || len == 0) return
        bindings.denseRotm(
            x, xOff, xStride, y, yOff, yStride, len,
            transformation.h11, transformation.h12, transformation.h21, transformation.h22,
        )
    }

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        if (len == 0) return
        bindings.denseRotm(x, xOff, 1, y, yOff, 1, len, c, s, -s, c)
    }

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (len == 0) return
        bindings.denseSwap(a, aOff, b, bOff, len)
    }

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double {
        if (len == 0) return 0.0
        return bindings.denseSum(v, vOff, len)
    }

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        if (len == 0) return 0.0
        return bindings.denseSsqd(a, aOff, b, bOff, len)
    }
}
