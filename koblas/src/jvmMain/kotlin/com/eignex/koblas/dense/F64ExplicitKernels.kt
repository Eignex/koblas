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
     * that cross at all. What was measured, and why only reductions appear here, is on
     * [DenseTuning.jvmCDotCrossover]. Bound to fields of this object so the comparison each routine makes
     * does not reach through the tuning object on every call. [F64SimdKernels] gates on vector width
     * instead, because its cost is a vector rather than a foreign call.
     */
    private val DOT_C_CROSSOVER = DenseTuning.jvmCDotCrossover
    private val SUM_C_CROSSOVER = DenseTuning.jvmCSumCrossover
    private val SSQD_C_CROSSOVER = DenseTuning.jvmCSsqdCrossover
    private val NRM2_C_CROSSOVER = DenseTuning.jvmCNrm2Crossover
    private val ASUM_C_CROSSOVER = DenseTuning.jvmCAsumCrossover
    private val DOT4_C_CROSSOVER = DenseTuning.jvmCDot4Crossover

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
    override val gemmTileRows: Int get() = if (simdAvailable) Simd.tileRows else super.gemmTileRows

    override val gemmTileCols: Int get() = if (simdAvailable) Simd.TILE_COLS else super.gemmTileCols

    @Suppress("LongParameterList")
    override fun gemmTile(
        depth: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
        ldc: Int,
    ) {
        if (simdAvailable) {
            Simd.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
        } else {
            super.gemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
        }
    }

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
