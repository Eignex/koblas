package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot
import com.eignex.koblas.portableRotm
import com.eignex.koblas.portableRotmg

/** The bundled C kernels without automatic SIMD selection. */
internal class CVectorKernels(private val bindings: JvmCKernelBindings, private val exact: Boolean) :
    DenseVectorKernels {
    /**
     * Run length from which crossing into the bundled library beats staying on the JVM, for the routines
     * that cross at all. What was measured, and why only reductions appear here, is on
     * [DenseTuning.jvmCDotCrossover]. Bound to fields of this object so the comparison each routine makes
     * does not reach through the tuning object on every call. [SimdVectorKernels] gates on vector width
     * instead, because its cost is a vector rather than a foreign call.
     */
    private val dotCCrossover = DenseTuning.jvmCDotCrossover
    private val sumCCrossover = DenseTuning.jvmCSumCrossover
    private val ssqdCCrossover = DenseTuning.jvmCSsqdCrossover
    private val nrm2CCrossover = DenseTuning.jvmCNrm2Crossover
    private val iamaxCCrossover = DenseTuning.jvmCIamaxCrossover
    private val asumCCrossover = DenseTuning.jvmCAsumCrossover

    override val name: String get() = "c-${bindings.variant.name.lowercase()}" + if (exact) "-raw" else "-policy"

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (!exact && len < dotCCrossover) {
            scalarDot(a, aOff, b, bOff, len)
        } else {
            bindings.denseDot(a, aOff, b, bOff, len)
        }

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (!exact && len < sumCCrossover) scalarSum(v, vOff, len) else bindings.denseSum(v, vOff, len)

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        if (!exact && len < ssqdCCrossover) {
            scalarSsqd(a, aOff, b, bOff, len)
        } else {
            bindings.denseSsqd(a, aOff, b, bOff, len)
        }

    // No CBLAS or C routine generates the modified Givens transformation, so the portable one is the
    // implementation rather than a fallback.
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens = portableRotmg(d1, d2, x1, y1)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        if (exact) bindings.denseAxpy(y, yOff, alpha, x, xOff, len) else scalarAxpy(y, yOff, alpha, x, xOff, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) =
        if (exact) bindings.denseScale(v, vOff, alpha, len) else scalarScale(v, vOff, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        if (!exact && len < nrm2CCrossover) euclideanNorm(v, vOff, len) else bindings.denseNrm2(v, vOff, len)

    override fun iamax(v: DoubleArray, vOff: Int, len: Int): Int =
        if (!exact && len < iamaxCCrossover) scalarIamax(v, vOff, len) else bindings.denseIamax(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        if (!exact && len < asumCCrossover) absoluteSum(v, vOff, len) else bindings.denseAsum(v, vOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        if (exact) bindings.denseSwap(a, aOff, b, bOff, len) else scalarSwap(a, aOff, b, bOff, len)

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
        if (exact) {
            bindings.denseRotm(
                x, xOff, xStride, y, yOff, yStride, len,
                transformation.h11, transformation.h12, transformation.h21, transformation.h22,
            )
        } else {
            portableRotm(x, xOff, xStride, y, yOff, yStride, len, transformation)
        }
    }

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        if (exact) {
            bindings.denseRotm(x, xOff, 1, y, yOff, 1, len, c, s, -s, c)
        } else {
            portableRot(x, xOff, y, yOff, len, c, s)
        }
}
