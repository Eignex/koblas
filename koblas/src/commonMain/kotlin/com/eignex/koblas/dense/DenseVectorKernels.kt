package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.internal.numeric.*
import com.eignex.koblas.portableRot
import com.eignex.koblas.portableRotm
import com.eignex.koblas.portableRotmg

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

    /** Returns `sum(a[aOff + i] * b[bOff + i])` over [len] entries, or zero for an empty run. */
    public fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

    /**
     * Adds `alpha * x[xOff + i]` into `y[yOff + i]` over [len] entries. A zero [alpha] returns without
     * evaluating the products, as required by standalone BLAS AXPY semantics.
     */
    public fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

    /** Scales [len] entries in [v] by [alpha]. */
    public fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)

    /**
     * Returns the rescaled Euclidean norm over [len] entries, or zero for an empty run. Implementations
     * must avoid the avoidable overflow and underflow of a plain square-sum followed by a square root.
     */
    public fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double

    /** Sum of absolute values over [len] entries, or zero for an empty run. */
    public fun asum(v: DoubleArray, vOff: Int, len: Int): Double

    /** Constructs a modified Givens transformation. */
    public fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens

    /**
     * Applies a modified Givens [transformation] to two strided runs. Each pair is loaded before either
     * result is stored. Equal runs are therefore safe, but callers must snapshot any other overlap whose
     * stores could precede a later input load.
     */
    @Suppress("LongParameterList")
    public fun rotm(
        x: DoubleArray,
        xOff: Int,
        xStride: Int,
        y: DoubleArray,
        yOff: Int,
        yStride: Int,
        len: Int,
        transformation: ModifiedGivens,
    )

    /**
     * Applies the plane rotation `(c, s, -s, c)` to two contiguous runs. Each pair is loaded before either
     * result is stored. Equal runs are therefore safe, but callers must snapshot any other overlap whose
     * stores could precede a later input load.
     */
    @Suppress("LongParameterList")
    public fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double)

    /** Exchanges [len] entries of the two runs. */
    public fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int)

    /**
     * Returns the plain sum over [len] entries, or zero for an empty run. This is not compensated summation;
     * compiled leaves may reduce vector lanes as a tree.
     */
    public fun sum(v: DoubleArray, vOff: Int, len: Int): Double

    /**
     * Returns `sum((a[aOff + i] - b[bOff + i])^2)` over [len] entries, or zero for an empty run. This is a
     * single-pass squared distance and may overflow; it does not inherit the rescaling contract of [nrm2].
     */
    public fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double
}

/** Pure Kotlin scalar kernels retained as the portable fallback and semantic reference for compiled leaves. */
internal object ScalarKernels : DenseVectorKernels {
    override val name: String get() = "scalar"

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        scalarDot(a, aOff, b, bOff, len)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        scalarAxpy(y, yOff, alpha, x, xOff, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) = scalarScale(v, vOff, alpha, len)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = euclideanNorm(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double = scalarSum(v, vOff, len)

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        scalarSsqd(a, aOff, b, bOff, len)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        scalarSwap(a, aOff, b, bOff, len)

    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens = portableRotmg(d1, d2, x1, y1)

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
