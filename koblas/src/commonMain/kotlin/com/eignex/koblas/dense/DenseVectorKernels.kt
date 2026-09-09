package com.eignex.koblas.dense

import com.eignex.koblas.ModifiedGivens

/**
 * Contiguous dense Level 1 operations that are independently useful outside matrix algorithms.
 * Implementations may differ in their final rounding because compiled leaves can fuse multiplication
 * and addition or reduce vector lanes as a tree. A zero [len] is legal for every operation.
 */
public interface DenseVectorKernels {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /** Sum of pairwise products over [len] entries, or zero for an empty run. */
    public fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

    /** Adds `alpha * x` into [y] over [len] entries; zero alpha follows standalone BLAS semantics. */
    public fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

    /** Scales [len] entries in [v] by [alpha]. */
    public fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)

    /** Rescaled Euclidean norm over [len] entries, or zero for an empty run. */
    public fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double

    /** Sum of absolute values over [len] entries, or zero for an empty run. */
    public fun asum(v: DoubleArray, vOff: Int, len: Int): Double

    /** Constructs a modified Givens transformation. */
    public fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens

    /** Applies a modified Givens [transformation] to two strided runs. */
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

    /** Applies the plane rotation `(c, s, -s, c)` to two contiguous runs. */
    @Suppress("LongParameterList")
    public fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double)

    /** Exchanges [len] entries of the two runs. */
    public fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int)

    /** Plain sum over [len] entries, or zero for an empty run. */
    public fun sum(v: DoubleArray, vOff: Int, len: Int): Double

    /** Sum of squared pairwise differences over [len] entries. */
    public fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double
}

/** The dense vector kernels selected once for the current platform. */
internal expect object PlatformVectorKernels : DenseVectorKernels {
    override val name: String
    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double
    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)
    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)
    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double
    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double
    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens

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
    )

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double)
    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int)
    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double
    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double
}

/** Platform packed family used by the consumer-facing packed-panel API. */
internal val platformPackedKernels: PackedKernels get() = platformDenseKernelFamilies.packed
