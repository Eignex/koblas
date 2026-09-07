package com.eignex.koblas.dense

import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.applyModifiedGivens
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot
import com.eignex.koblas.portableRotm
import com.eignex.koblas.portableRotmg
import kotlin.math.sqrt

/** The bundled C kernels without automatic SIMD selection. */
internal object F64CKernels : F64Kernels, F64ArithmeticKernels {
    /**
     * Run length from which crossing into the bundled library beats staying on the JVM, for the routines
     * that cross at all. Every call here wraps each array in a MemorySegment and goes through invokeExact,
     * which costs tens of nanoseconds whatever the length, so a short run pays for a foreign call to do
     * work the JIT would have finished already.
     *
     * Measured on `Level1Benchmark` with these constants temporarily set to zero, so the C arm really
     * crosses rather than falling back to the same portable code the scalar arm runs. Across three runs the
     * four plain reductions are level or behind at 64 and ahead at 128, and ssqd, which reads two operands,
     * only separates at 256. Past its crossover each pulls away, reaching 2.8x to 4.1x by 2048.
     *
     * Only reductions appear here. HotSpot will not vectorise a floating-point reduction, since splitting
     * the sum across lanes reorders the additions and changes the result, so those loops run an element at
     * a time however hot they get and the C kernel has something to beat. That is the whole of what the
     * bundled library wins on this platform, and [scale] covers the elementwise routines that never cross.
     *
     * Per-operation constants, as the host crossovers in [F64RoutedKernels] are, so a later measurement can
     * move one alone. [F64SimdKernels] gates on vector width instead, because its cost is a vector rather
     * than a foreign call.
     */
    private const val DOT_C_CROSSOVER = 128
    private const val SUM_C_CROSSOVER = 128
    private const val SSQD_C_CROSSOVER = 256
    private const val NRM2_C_CROSSOVER = 128
    private const val ASUM_C_CROSSOVER = 128

    /**
     * Four dots share one pass over the shared operand, so the same foreign call covers four runs of this
     * length. The point estimate leads from 256 upward and holds about 1.4x, but the JIT arm varies enough
     * that 512 is the shortest length where the two separate beyond their error bars, which is the same
     * criterion the other crossovers use.
     *
     * This is the one kernel the AVX2 clones in `koblas_kernels.h` leave alone, since the wider registers
     * cost it 1.7x, so what it crosses into here is the baseline build.
     */
    private const val DOT4_C_CROSSOVER = 512

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
        scalarAxpy(y, yOff, alpha, x, xOff, len)

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)

    /**
     * Portable at every length rather than past a crossover, as [axpy], [swap], [rot] and [rotm] are.
     *
     * None of them carries a reduction, so HotSpot vectorises them on its own, and the loop it produces is
     * wider than the one in the bundled library: that is compiled without a target flag and emits two-wide
     * SSE2 where the JIT emits four-wide AVX2. `Level1Benchmark` with the crossovers set to zero puts the C
     * kernel behind at every length from 16 to 2048, still 1.7x to 3x behind at the top and not
     * converging, so no length would repay the crossing and there is no constant to measure. Building the
     * C for a wider instruction set would change that and would have to be measured again.
     */
    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) = scalarScale(v, vOff, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len < NRM2_C_CROSSOVER) euclideanNorm(v, vOff, len) else JvmCKernelBindings.denseNrm2(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len < ASUM_C_CROSSOVER) absoluteSum(v, vOff, len) else JvmCKernelBindings.denseAsum(v, vOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        scalarSwap(a, aOff, b, bOff, len)

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
    ) = if (len < DOT4_C_CROSSOVER) {
        scalarDot4(a, aOff, stride, b, bOff, len, out, outOff)
    } else {
        JvmCKernelBindings.denseDot4(a, aOff, stride, b, bOff, len, out, outOff)
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
    ) = portableRotm(x, xOff, xStride, y, yOff, yStride, len, transformation)

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        portableRot(x, xOff, y, yOff, len, c, s)
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
