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

/**
 * The JVM Vector API kernels, which are the reductions and nothing else.
 *
 * A reduction is what the compiler cannot do for us. Vectorising a sum of doubles means adding them in a
 * different order, which changes the answer, so HotSpot leaves the loop alone unless it is written in lanes.
 * Writing it in lanes is worth 3 to 10 times the scalar loop for `dot`, `asum` and `sum` above 128.
 *
 * Elementwise work is the opposite: each element's result depends on nothing else, so the JIT vectorises the
 * ordinary Kotlin loop by itself, and there is nothing here for `axpy`, `scal`, `swap` or `rot`. Written in
 * lanes they measure 0.93 to 1.10 against the portable loops, and `scal` measures 0.44 to 0.55 above 512.
 * Measured on 12th Gen Intel Core i9-12900H with the sweep suite, comparing the `jvm-scalar` and `jvm-simd`
 * targets.
 *
 * `nrm2` is here because its fast path is a sum of squares, which is a reduction; it falls back to the
 * rescaling loop when a value leaves the normal range.
 */
internal object SimdVectorKernels : DenseVectorKernels {
    /**
     * Width from which the vector search for the first largest magnitude beats the scalar one.
     *
     * A fixed measured constant rather than a tuning key. The index search carries a lane-position vector
     * beside the magnitude one and reduces both, so it pays later than the plain reductions do.
     *
     * This constant gates its own comparison: at this value every narrower run takes the scalar kernel in
     * both arms, so measuring it means lowering it to the lane width and running the `iamax` sweep over the
     * `jvm-scalar` and `jvm-simd` targets. Doing that on an i9-12900H has the vectorised search losing below
     * 64, inside the noise from 64 to 192, losing outright at 128, and ahead at every width from 256 upward
     * by 1.13 to 1.90 — which is where it starts winning and staying ahead.
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
        -> if (vectorizes(length)) name else ScalarVectorKernels.name

        // The elementwise four have no vector kernel here; see the class documentation.
        DenseOperation.Axpy, DenseOperation.Scale, DenseOperation.Swap, DenseOperation.Rot,
        -> ScalarVectorKernels.name
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
    ) = scalarAxpy(y, yOff, yStride, alpha, x, xOff, xStride, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) =
        scalarScale(v, vOff, vStride, alpha, len)

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

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        scalarSwap(a, aOff, aStride, b, bOff, bStride, len)

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        portableRot(x, xOff, y, yOff, len, c, s)
}
