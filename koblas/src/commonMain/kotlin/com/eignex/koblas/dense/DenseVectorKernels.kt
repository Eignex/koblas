package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot

/**
 * Contiguous dense Level 1 operations that are independently useful outside matrix algorithms.
 * Implementations must read and write only the array windows selected by their offsets and lengths.
 * Callers must supply non-negative lengths and valid windows; implementations need not validate them.
 * A zero length is legal for every operation that accepts one and produces the documented empty result
 * without reading either array.
 *
 * Results agree to within rounding rather than bit for bit: compiled leaves may fuse multiplication and
 * addition or reduce vector lanes as a tree. Shared algorithms must not depend on the final rounding order.
 */
public interface DenseVectorKernels {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /**
     * The implementation a call of this [operation] and [length] reaches, or null when its own values decide.
     *
     * [name] identifies a selection, which is not the same question: a selection that dispatches by length
     * answers with one of its components, and a caller that reports the selection name has named something
     * that did not run. An implementation which does not dispatch answers with itself, which is the default.
     *
     * Null is the honest answer where no width establishes the destination, as for a norm that retries through
     * a rescaling loop when the square sum leaves the normal range. Such a call is not an exact measurement of
     * either kernel, and asking after the fact would mean tracing inside the timed region.
     */
    public fun implementationFor(operation: DenseOperation, length: Int, contiguous: Boolean = true): String? = name

    /** Returns `sum(a[aOff + i] * b[bOff + i])` over [len] entries, or zero for an empty run. */
    public fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Double

    /**
     * Adds `alpha * x[xOff + i]` into `y[yOff + i]` over [len] entries. A zero [alpha] returns without
     * evaluating the products, as required by standalone BLAS AXPY semantics.
     */
    public fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int = 1,
        xStride: Int = 1,
    )

    /** Scales [len] entries in [v] by [alpha]. */
    public fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int = 1)

    /**
     * Returns the rescaled Euclidean norm over [len] entries, or zero for an empty run. Implementations
     * must avoid the avoidable overflow and underflow of a plain square-sum followed by a square root.
     */
    public fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double

    /** Sum of absolute values over [len] entries, or zero for an empty run. */
    public fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double

    /**
     * Zero-based index within the selected run of a maximum absolute value, or `-1` when empty. A nonempty
     * run of zeros returns `0`.
     *
     * Which index is returned when the maximum is not unique, and what happens when the largest magnitude is
     * a NaN, is this implementation's. `idamax` specifies neither: the portable kernels compare strictly, so
     * ties go to the first and a NaN loses to a later finite entry, while a vendor may report the NaN's index
     * and may resolve a tie the other way for a negatively spaced run.
     */
    public fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Int

    /**
     * Applies the plane rotation `(c, s, -s, c)` to two contiguous runs. Each pair is loaded before either
     * result is stored. Equal runs are therefore safe, but callers must snapshot any other overlap whose
     * stores could precede a later input load.
     */
    @Suppress("LongParameterList")
    public fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double)

    /** Exchanges [len] entries of the two runs. */
    public fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int = 1, bStride: Int = 1)

    /**
     * Returns the plain sum over [len] entries, or zero for an empty run. This is not compensated summation;
     * compiled leaves may reduce vector lanes as a tree.
     */
    public fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double
}

/** Pure Kotlin scalar kernels retained as the portable fallback and semantic reference for compiled leaves. */
internal object ScalarVectorKernels : DenseVectorKernels {
    override val name: String get() = "scalar"

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double = scalarDot(a, aOff, aStride, b, bOff, bStride, len)

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

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = euclideanNorm(v, vOff, vStride, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = absoluteSum(v, vOff, vStride, len)

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int = scalarIamax(v, vOff, vStride, len)

    override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double = scalarSum(v, vOff, vStride, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        scalarSwap(a, aOff, aStride, b, bOff, bStride, len)

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        portableRot(x, xOff, y, yOff, len, c, s)
}
