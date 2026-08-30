package com.eignex.koblas.dense

import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.internal.numeric.*
import kotlin.math.sqrt

/** The bundled C kernels without automatic SIMD selection. */
internal object F64CKernels : F64Kernels {
    override val name: String get() = BackendNames.C

    override val isPortable: Boolean get() = true

    override val isAvailable: Boolean get() = JvmCKernelBindings.isAvailable

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        JvmCKernelBindings.denseDot(a, aOff, b, bOff, len)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        JvmCKernelBindings.denseAxpy(y, yOff, alpha, x, xOff, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) =
        JvmCKernelBindings.denseScale(v, vOff, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = JvmCKernelBindings.denseNrm2(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = JvmCKernelBindings.denseAsum(v, vOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        JvmCKernelBindings.denseSwap(a, aOff, b, bOff, len)

    @Suppress("LongParameterList")
    override fun symvColumn(
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: Double,
        len: Int,
    ): Double = JvmCKernelBindings.denseSymvColumn(a, aOff, x, xOff, y, yOff, mult, len)

    @Suppress("LongParameterList")
    override fun symvColumn4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: DoubleArray,
        out: DoubleArray,
        len: Int,
    ) = JvmCKernelBindings.denseSymvColumn4(a, aOff, stride, x, xOff, y, yOff, mult, out, len)

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
    ) = JvmCKernelBindings.denseDot4(a, aOff, stride, b, bOff, len, out, outOff)
}

/** The JVM Vector API kernels without automatic C selection. */
internal object F64SimdKernels : F64Kernels {
    private val lanes: Int = if (simdAvailable) Simd.lanes() else 0

    override val name: String get() = "${BackendNames.SIMD}($lanes lanes)"

    override val isPortable: Boolean get() = true

    override val isAvailable: Boolean get() = simdAvailable

    private fun vectorizes(len: Int): Boolean = simdAvailable && len >= lanes

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (vectorizes(len)) Simd.dot(a, aOff, b, bOff, len) else scalarDot(a, aOff, b, bOff, len)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (vectorizes(len)) Simd.axpy(y, yOff, alpha, x, xOff, len) else scalarAxpy(y, yOff, alpha, x, xOff, len)
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (vectorizes(len)) Simd.scale(v, vOff, alpha, len) else scalarScale(v, vOff, alpha, len)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double {
        if (vectorizes(len)) {
            val squares = Simd.dot(v, vOff, v, vOff, len)
            if (squares.isFinite() && squares >= F64_MIN_NORMAL) return sqrt(squares)
        }
        return euclideanNorm(v, vOff, len)
    }

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (vectorizes(len)) Simd.asum(v, vOff, len) else absoluteSum(v, vOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        if (vectorizes(len)) Simd.swap(a, aOff, b, bOff, len) else super.swap(a, aOff, b, bOff, len)
    }

    @Suppress("LongParameterList")
    override fun symvColumn(
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: Double,
        len: Int,
    ): Double = if (vectorizes(len)) {
        Simd.symvColumn(a, aOff, x, xOff, y, yOff, mult, len)
    } else {
        super.symvColumn(a, aOff, x, xOff, y, yOff, mult, len)
    }

    @Suppress("LongParameterList")
    override fun symvColumn4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: DoubleArray,
        out: DoubleArray,
        len: Int,
    ) {
        if (vectorizes(len)) {
            Simd.symvColumn4(a, aOff, stride, x, xOff, y, yOff, mult, out, len)
        } else {
            super.symvColumn4(a, aOff, stride, x, xOff, y, yOff, mult, out, len)
        }
    }

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
    ) {
        if (vectorizes(len)) {
            Simd.dot4(a, aOff, stride, b, bOff, len, out, outOff)
        } else {
            scalarDot4(a, aOff, stride, b, bOff, len, out, outOff)
        }
    }
}
