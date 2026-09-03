package com.eignex.koblas.dense

import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.applyModifiedGivens
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot
import com.eignex.koblas.portableRotmg
import kotlin.math.sqrt

/** The bundled C kernels without automatic SIMD selection. */
internal object F64CKernels : F64Kernels, F64ArithmeticKernels {
    /**
     * Run length from which crossing into the bundled library beats staying on the JVM. Every call here
     * wraps each array in a MemorySegment and goes through invokeExact, which costs tens of nanoseconds
     * whatever the length, so a short run pays for a foreign call to do work the JIT would have finished
     * already.
     *
     * GemvIntoBenchmark shows what an ungated boundary costs a caller that drives one short run per stored
     * entry: 515 ns against 60 ns at two rows. Level1Benchmark's c and scalar runs put the crossover
     * between 64 and 128 elements, the scalar loop still winning at 64 for dot and nrm2 and the C kernel
     * winning outright by 128, so these sit at 128. The asymmetry argues for the higher end: crossing too
     * early costs 30 to 50 ns a call, staying too long costs about 10.
     *
     * This is a different number from [F64SimdKernels]'s lane check, which gates on vector width because
     * its cost is vector width rather than a foreign call. Per-operation constants, as
     * [F64RoutedKernels]'s host crossovers are, so a later measurement can move one alone.
     */
    private const val DOT_C_CROSSOVER = 128
    private const val SUM_C_CROSSOVER = 128
    private const val SSQD_C_CROSSOVER = 128
    private const val AXPY_C_CROSSOVER = 128
    private const val SCALE_C_CROSSOVER = 128
    private const val NRM2_C_CROSSOVER = 128
    private const val ASUM_C_CROSSOVER = 128
    private const val SWAP_C_CROSSOVER = 128

    override val name: String get() = BackendNames.C

    override val isPortable: Boolean get() = true

    override val isAvailable: Boolean get() = JvmCKernelBindings.isAvailable

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (len < DOT_C_CROSSOVER) {
            scalarDot(a, aOff, b, bOff, len)
        } else {
            JvmCKernelBindings.denseDot(a, aOff, b, bOff, len)
        }

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len < SUM_C_CROSSOVER) scalarSum(v, vOff, len) else JvmCKernelBindings.denseSum(v, vOff, len)

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (len < SSQD_C_CROSSOVER) {
            scalarSsqd(a, aOff, b, bOff, len)
        } else {
            JvmCKernelBindings.denseSsqd(a, aOff, b, bOff, len)
        }

    // No CBLAS or C routine generates the modified Givens transformation, so the portable one is the
    // implementation rather than a fallback.
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens =
        portableRotmg(d1, d2, x1, y1)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        if (len < AXPY_C_CROSSOVER) {
            scalarAxpy(y, yOff, alpha, x, xOff, len)
        } else {
            JvmCKernelBindings.denseAxpy(y, yOff, alpha, x, xOff, len)
        }

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        if (len < AXPY_C_CROSSOVER) {
            scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
        } else {
            JvmCKernelBindings.denseAxpyArithmetic(y, yOff, alpha, x, xOff, len)
        }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) = if (len < SCALE_C_CROSSOVER) {
        scalarScale(v, vOff, alpha, len)
    } else {
        JvmCKernelBindings.denseScale(v, vOff, alpha, len)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len < NRM2_C_CROSSOVER) euclideanNorm(v, vOff, len) else JvmCKernelBindings.denseNrm2(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len < ASUM_C_CROSSOVER) absoluteSum(v, vOff, len) else JvmCKernelBindings.denseAsum(v, vOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) = if (len < SWAP_C_CROSSOVER) {
        scalarSwap(a, aOff, b, bOff, len)
    } else {
        JvmCKernelBindings.denseSwap(a, aOff, b, bOff, len)
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
    ) = JvmCKernelBindings.denseDot4(a, aOff, stride, b, bOff, len, out, outOff)

    @Suppress("LongParameterList")
    override fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: F64ModifiedGivens,
    ) {
        if (transformation.flag == -2.0) return
        JvmCKernelBindings.denseRotm(
            x,
            xOff,
            xStride,
            y,
            yOff,
            yStride,
            len,
            transformation.h11,
            transformation.h12,
            transformation.h21,
            transformation.h22,
        )
    }

    // A plane rotation is the modified Givens transformation (c, s, -s, c), so it goes to the same kernel.
    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        JvmCKernelBindings.denseRotm(x, xOff, 1, y, yOff, 1, len, c, s, -s, c)
}

/** The JVM Vector API kernels without automatic C selection. */
internal object F64SimdKernels : F64Kernels, F64ArithmeticKernels {
    private val lanes: Int = if (simdAvailable) Simd.lanes() else 0

    override val name: String get() = "${BackendNames.SIMD}($lanes lanes)"

    override val isPortable: Boolean get() = true

    override val isAvailable: Boolean get() = simdAvailable

    private fun vectorizes(len: Int): Boolean = simdAvailable && len >= lanes

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (vectorizes(len)) Simd.dot(a, aOff, b, bOff, len) else scalarDot(a, aOff, b, bOff, len)

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (vectorizes(len)) Simd.sum(v, vOff, len) else scalarSum(v, vOff, len)

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double = if (vectorizes(len)) {
        Simd.ssqd(a, aOff, b, bOff, len)
    } else {
        scalarSsqd(a, aOff, b, bOff, len)
    }

    // No CBLAS or C routine generates the modified Givens transformation, so the portable one is the
    // implementation rather than a fallback.
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens =
        portableRotmg(d1, d2, x1, y1)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (vectorizes(len)) Simd.axpy(y, yOff, alpha, x, xOff, len) else scalarAxpy(y, yOff, alpha, x, xOff, len)
    }

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (vectorizes(len)) {
            Simd.axpyArithmetic(y, yOff, alpha, x, xOff, len)
        } else {
            scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
        }
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
        if (vectorizes(len)) Simd.swap(a, aOff, b, bOff, len) else scalarSwap(a, aOff, b, bOff, len)
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

    @Suppress("LongParameterList")
    override fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: F64ModifiedGivens,
    ) {
        if (transformation.flag == -2.0) return
        if (vectorizes(len) && xStride == 1 && yStride == 1) {
            Simd.rotm(
                x,
                xOff,
                y,
                yOff,
                len,
                transformation.h11,
                transformation.h12,
                transformation.h21,
                transformation.h22,
            )
        } else {
            applyModifiedGivens(x, xOff, xStride, y, yOff, yStride, len, transformation)
        }
    }

    // A plane rotation is the modified Givens transformation (c, s, -s, c), so it goes to the same kernel.
    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        if (vectorizes(len)) {
            Simd.rotm(x, xOff, y, yOff, len, c, s, -s, c)
        } else {
            portableRot(x, xOff, y, yOff, len, c, s)
        }
    }
}
