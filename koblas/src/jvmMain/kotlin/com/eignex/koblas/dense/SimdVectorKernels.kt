package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot
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
     * beside the magnitude one and reduces both, so it pays later than the plain reductions do.
     *
     * Measured, and unchanged by the measurement. This constant gates its own comparison: at the shipped
     * value every narrower run takes the scalar kernel in both arms, so the two were timed once more with it
     * lowered to the lane width. The vectorised search loses below 64, sits inside the noise from 64 to 192
     * and loses outright at 128, and from 256 is ahead at every width measured, by 1.13 to 1.90. The capture
     * is in `koblas-bench/reports/level1-crossover/jvm-iamax-gate-lowered/`, on the host named beside it.
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

    // A plane rotation is two fused multiply-adds per lane, which is what the vector kernel below does.
    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
        if (vectorizes(len)) {
            SimdOps.rot(x, xOff, y, yOff, len, c, s)
        } else {
            portableRot(x, xOff, y, yOff, len, c, s)
        }
    }
}
