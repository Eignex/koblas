package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot
import com.eignex.koblas.portableRotm
import com.eignex.koblas.portableRotmg

/** The bundled C kernels without automatic SIMD selection. */
internal object CVectorKernels : DenseVectorKernels {
    /**
     * Run length from which crossing into the bundled library beats staying on the JVM, for the routines
     * that cross at all. What was measured, and why only reductions appear here, is on
     * [DenseTuning.jvmCDotCrossover]. Bound to fields of this object so the comparison each routine makes
     * does not reach through the tuning object on every call. [SimdVectorKernels] gates on vector width
     * instead, because its cost is a vector rather than a foreign call.
     */
    private val DOT_C_CROSSOVER = DenseTuning.jvmCDotCrossover
    private val SUM_C_CROSSOVER = DenseTuning.jvmCSumCrossover
    private val SSQD_C_CROSSOVER = DenseTuning.jvmCSsqdCrossover
    private val NRM2_C_CROSSOVER = DenseTuning.jvmCNrm2Crossover
    private val IAMAX_C_CROSSOVER = DenseTuning.jvmCIamaxCrossover
    private val ASUM_C_CROSSOVER = DenseTuning.jvmCAsumCrossover

    override val name: String get() = "c"

    val isAvailable: Boolean get() = JvmCKernelBindings.isAvailable

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
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens = portableRotmg(d1, d2, x1, y1)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        scalarAxpy(y, yOff, alpha, x, xOff, len)

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

    override fun iamax(v: DoubleArray, vOff: Int, len: Int): Int =
        if (len < IAMAX_C_CROSSOVER) scalarIamax(v, vOff, len) else JvmCKernelBindings.denseIamax(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (len < ASUM_C_CROSSOVER) absoluteSum(v, vOff, len) else JvmCKernelBindings.denseAsum(v, vOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        scalarSwap(a, aOff, b, bOff, len)

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
    ) = portableRotm(x, xOff, xStride, y, yOff, yStride, len, transformation)

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        portableRot(x, xOff, y, yOff, len, c, s)
}
