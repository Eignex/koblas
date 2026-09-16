package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.applyModifiedGivens
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot
import com.eignex.koblas.portableRotmg
import kotlin.math.sqrt

/** Whether the incubating Vector API resolved without initializing its implementation classes. */
internal val simdAvailable: Boolean = try {
    Class.forName("jdk.incubator.vector.DoubleVector")
    true
} catch (_: Throwable) {
    false
}

/** The JVM Vector API kernels without automatic C selection. */
internal object SimdVectorKernels : DenseVectorKernels {
    /**
     * Width from which the vector search for the first largest magnitude beats the scalar one.
     *
     * A fixed measured constant rather than a tuning key. The index search carries a lane-position vector
     * beside the magnitude one and reduces both, so it pays later than the plain reductions do; K4 revisits
     * this against the new measurements, which is where a crossover is allowed to move.
     */
    private const val IAMAX_CROSSOVER = 256

    private val lanes: Int = if (simdAvailable) SimdOps.lanes() else 0

    override val name: String get() = "simd($lanes lanes)"

    val isAvailable: Boolean get() = simdAvailable

    private fun vectorizes(len: Int): Boolean = simdAvailable && len >= lanes

    /**
     * Whether a run of this width and spacing reaches a vector kernel.
     *
     * The Vector API loads a lane block from consecutive elements, so a stride other than one would have to be
     * gathered. On this hardware that loses: the same measurement that keeps `SparseSimd` off AVX2 gathers
     * applies here, and a gather-backed dot would be slower than the scalar loop it replaced. A strided run is
     * therefore scalar work, and [implementationFor] says so rather than letting the selection name imply
     * otherwise.
     */
    private fun vectorizes(len: Int, contiguous: Boolean): Boolean = contiguous && vectorizes(len)

    override fun implementationFor(operation: DenseOperation, length: Int, contiguous: Boolean): String? {
        if (!contiguous) return ScalarVectorKernels.name
        return vectorisedImplementationFor(operation, length)
    }

    private fun vectorisedImplementationFor(operation: DenseOperation, length: Int): String? = when (operation) {
        // The square sum is tried first and abandoned for the rescaling loop when it leaves the normal range,
        // so which kernel produces the norm depends on the values rather than on the width.
        DenseOperation.Nrm2 -> if (vectorizes(length)) null else ScalarVectorKernels.name

        // Both strides must be one, and a flagged identity returns without arithmetic, so a width is not enough.
        DenseOperation.Rotm -> null

        DenseOperation.Iamax ->
            if (vectorizes(length) && length >= IAMAX_CROSSOVER) name else ScalarVectorKernels.name

        DenseOperation.Dot, DenseOperation.Sum, DenseOperation.Asum,
        DenseOperation.Axpy, DenseOperation.Scale, DenseOperation.Swap, DenseOperation.Rot,
        -> if (vectorizes(length)) name else ScalarVectorKernels.name
    }

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = if (vectorizes(len, aStride == 1 && bStride == 1)) {
        SimdOps.dot(a, aOff, b, bOff, len)
    } else {
        scalarDot(a, aOff, aStride, b, bOff, bStride, len)
    }

    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        if (vectorizes(len, vStride == 1)) SimdOps.sum(v, vOff, len) else scalarSum(v, vOff, vStride, len)

    // No CBLAS or C routine generates the modified Givens transformation, so the portable one is the
    // implementation rather than a fallback.
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens = portableRotmg(d1, d2, x1, y1)

    @Suppress("LongParameterList")
    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) {
        if (vectorizes(len, yStride == 1 && xStride == 1)) {
            SimdOps.axpy(y, yOff, alpha, x, xOff, len)
        } else {
            scalarAxpy(y, yOff, yStride, alpha, x, xOff, xStride, len)
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        if (vectorizes(
                len,
                vStride == 1,
            )
        ) {
            SimdOps.scale(v, vOff, alpha, len)
        } else {
            scalarScale(v, vOff, vStride, alpha, len)
        }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        if (vectorizes(len, vStride == 1)) {
            val squares = SimdOps.dot(v, vOff, v, vOff, len)
            if (squares.isFinite() && squares >= MIN_NORMAL) return sqrt(squares)
        }
        return euclideanNorm(v, vOff, vStride, len)
    }

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int =
        if (vectorizes(len, vStride == 1) && len >= IAMAX_CROSSOVER) {
            SimdOps.iamax(v, vOff, len)
        } else {
            scalarIamax(v, vOff, vStride, len)
        }

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        if (vectorizes(len, vStride == 1)) SimdOps.asum(v, vOff, len) else absoluteSum(v, vOff, vStride, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        if (vectorizes(len, aStride == 1 && bStride == 1)) {
            SimdOps.swap(a, aOff, b, bOff, len)
        } else {
            scalarSwap(a, aOff, aStride, b, bOff, bStride, len)
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
        transformation: ModifiedGivens,
    ) {
        if (transformation.flag == -2.0) return
        if (vectorizes(len) && xStride == 1 && yStride == 1) {
            SimdOps.rotm(
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
            SimdOps.rotm(x, xOff, y, yOff, len, c, s, -s, c)
        } else {
            portableRot(x, xOff, y, yOff, len, c, s)
        }
    }
}
