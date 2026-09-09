package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.ModifiedGivens

/**
 * The vector-vector routines as a backend half beneath [Blas]. Implementations must
 * agree with [PlatformKernels] to within rounding and read nothing outside the (offset, length)
 * window.
 *
 * To within rounding rather than exactly, because bit-for-bit is not a contract these routines can hold:
 * [PlatformKernels] itself fuses its multiply-add above one lane width and does not below it, and
 * reduces over lanes as a tree rather than in order. Two conforming implementations can differ in the last
 * bits of a sum, and the reference routines are written not to depend on which one they got.
 *
 * A length of zero is legal everywhere and does nothing: the triangular and Householder kernels reach the
 * last row with an empty tail, so every routine here is called that way.
 */
public interface Kernels : Backend {
    /** Sum of a(aOff + i) * b(bOff + i) over the first [len] entries; `0` for an empty run. */
    public fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

    /** Adds alpha * x(xOff + i) into y(yOff + i) over the first [len] entries. */
    public fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

    /** Scales the [len] entries from vOff by alpha. */
    public fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)

    /**
     * Euclidean norm of the [len] entries from vOff (BLAS `dnrm2`), `0` for an empty run. Must rescale to
     * stay in range, so a plain `sqrt(sum of squares)` is not a valid implementation.
     */
    public fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double

    /** Sum of the absolute values of the [len] entries from vOff (BLAS `dasum`); `0` for an empty run. */
    public fun asum(v: DoubleArray, vOff: Int, len: Int): Double

    /** Construct a modified Givens transformation (BLAS `drotmg`). */
    public fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens

    /**
     * Apply a modified Givens [transformation] (BLAS `drotm`) to [len] entries with independent offsets
     * and strides. Each pair is loaded before either result is stored; callers whose runs overlap must
     * snapshot them before calling this backend primitive.
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
     * Apply a plane rotation (BLAS `drot`) to the [len] pairs from the two offsets, replacing
     * `(x_i, y_i)` with `(c*x_i + s*y_i, c*y_i - s*x_i)`. Each pair is loaded before either result is
     * stored, which is what lets one read of a pair produce both of its outputs; callers whose runs
     * overlap must snapshot them before calling this backend primitive.
     *
     * Offsets and a length rather than strides, since the callers rotate contiguous runs.
     *
     * This is [rotm] at `(h11, h12, h21, h22) = (c, s, -s, c)`, so a leaf with a modified-Givens kernel
     * already has this one and implements it there rather than falling back to a loop of its own.
     */
    @Suppress("LongParameterList")
    public fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double)

    /**
     * Exchange the two runs (BLAS `dswap`). Two loads and two stores an element, so an implementation is
     * bound by memory rather than by issue rate, and a leaf with nothing better says so by calling the
     * portable loop itself.
     */
    public fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int)

    /**
     * Four dots against a shared right operand. For r in 0..3, out(outOff + r) is the dot of the run at
     * aOff + r * stride with the run at bOff. Defaults to four [dot] calls, which is the one default here
     * that costs nothing: each of the four reaches the same accelerated kernel a caller would have used.
     * An implementation that can read the shared operand once for all four should still override it.
     */
    @Suppress("LongParameterList") // four column offsets plus the shared operand
    public fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        for (r in 0 until 4) out[outOff + r] = dot(a, aOff + r * stride, b, bOff, len)
    }

    /**
     * Plain sum over the run, `0` for an empty one.
     *
     * Not a BLAS routine: `dasum` sums absolute values, which is a different quantity. Compensated
     * summation is deliberately not this routine, since a compensator split across vector lanes has a
     * different error bound from one carried in order; the public `compensatedSum` holds that contract.
     */
    public fun sum(v: DoubleArray, vOff: Int, len: Int): Double

    /**
     * Sum of squares of the differences, `sum (a(aOff + i) - b(bOff + i))^2`, in one pass over both runs;
     * `0` for an empty run. This is the squared euclidean distance between them.
     *
     * Not a BLAS routine, and not a rescaling one either: unlike [nrm2] this overflows once a difference
     * squares past the double range, which is what buys the single pass. It is also not `a.a - 2a.b + b.b`,
     * whose cancellation ruins exactly the small distances a nearest-neighbour search compares.
     *
     * The load pattern is [dot]'s with one subtract fused in, which is where an implementation with vector
     * units earns its keep. No BLAS library has this routine, so a leaf backed by one implements it over
     * whichever kernels it does have rather than inheriting anything.
     */
    public fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

    /**
     * Rows of C the matrix-product tile covers, which is how many contiguous doubles each step of the left
     * panel holds in [gemmTile]. Must be positive, and is fixed for the life of the backend, because the
     * caller packs its panels to this shape before calling.
     */
    public val gemmTileRows: Int get() = PORTABLE_TILE

    /** Columns of C the matrix-product tile covers, the width of each step of the right panel. */
    public val gemmTileCols: Int get() = PORTABLE_TILE

    /**
     * Adds one packed matrix product into a [gemmTileRows] by [gemmTileCols] tile of C.
     *
     * [packedA] holds [depth] groups of [gemmTileRows] contiguous doubles from [aOff], one group per step
     * of the shared dimension, and [packedB] holds [depth] groups of [gemmTileCols] from [bOff]. The tile
     * of C starts at [cOff] and its columns are [ldc] apart. Every group is full: the caller pads a short
     * edge with zeroes and discards what it did not want, so an implementation never sees a partial tile.
     *
     * This is the two-dimensional counterpart of [dot4]. Holding a tile rather than a row is the point: C
     * stays in registers across the whole of [depth], so it is read and written once per tile instead of
     * once per step, which is the difference between a product bounded by cache traffic and one bounded by
     * arithmetic. The default below is a plain four by four written out in scalars, which is correct
     * everywhere and is what a target without vector registers runs; every target that has them overrides
     * this with its own tile, and that override is the only place the matrix product varies by platform.
     */
    @Suppress("LongParameterList") // two packed panels, a destination tile, and the shared depth
    public fun gemmTile(
        depth: Int,
        packedA: DoubleArray,
        aOff: Int,
        packedB: DoubleArray,
        bOff: Int,
        c: DoubleArray,
        cOff: Int,
        ldc: Int,
    ) {
        portableGemmTile(depth, packedA, aOff, packedB, bOff, c, cOff, ldc)
    }
}

/** Internal vector leaf for parent routines whose arithmetic does not have DAXPY's zero-scalar return. */
internal interface ArithmeticKernels {
    @Suppress("LongParameterList")
    fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)
}

/**
 * The kernels compiled into this target: C on Native and on a JVM without `jdk.incubator.vector`, SIMD on
 * a JVM with the module. Its [Backend.name] is what `mathBackend` reports.
 */
internal expect object PlatformKernels : Kernels, ArithmeticKernels {
    override val name: String

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int)

    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double

    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

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

    @Suppress("LongParameterList") // four column offsets plus the shared operand
    override fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    )
}
