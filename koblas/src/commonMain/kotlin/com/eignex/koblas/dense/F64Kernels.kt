package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.F64ModifiedGivens

/**
 * The vector-vector routines as a backend half, alongside [F64Blas] and [F64Decompositions]. Implementations must
 * agree with [F64PlatformKernels] to within rounding and read nothing outside the (offset, length)
 * window.
 *
 * To within rounding rather than exactly, because bit-for-bit is not a contract these routines can hold:
 * [F64PlatformKernels] itself fuses its multiply-add above one lane width and does not below it, and
 * reduces over lanes as a tree rather than in order. Two conforming implementations can differ in the last
 * bits of a sum, and the reference routines are written not to depend on which one they got.
 *
 * A length of zero is legal everywhere and does nothing: the triangular and Householder kernels reach the
 * last row with an empty tail, so every routine here is called that way.
 */
public interface F64Kernels : Backend {
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
    public fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens

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
        transformation: F64ModifiedGivens,
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
}

/** Internal vector leaf for parent routines whose arithmetic does not have DAXPY's zero-scalar return. */
internal interface F64ArithmeticKernels {
    @Suppress("LongParameterList")
    fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)
}

/**
 * The kernels compiled into this target: C on Native and on a JVM without `jdk.incubator.vector`, SIMD on
 * a JVM with the module. Its [Backend.name] is what `mathBackend` reports.
 */
internal expect object F64PlatformKernels : F64Kernels, F64ArithmeticKernels {
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

    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens

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

/**
 * Uses a registered host backend when present, above its per-operation crossover. The `alpha` guards live
 * here, so `axpy` by zero and `scale` by one are no-ops whichever kernel runs.
 *
 * @property host a registered backend; null uses the compiled-in kernels, as does a run below the crossover.
 */
internal class F64RoutedKernels(internal val host: F64Kernels?) :
    F64Kernels,
    F64ArithmeticKernels {
    override val name: String
        get() = if (host == null) F64PlatformKernels.name else "${F64PlatformKernels.name}+${host.name}"

    // These describe the host this class was given, not what any one call routes to below its crossover:
    // a run under the crossover still executes on the compiled-in kernels even when host is non-null. Left
    // to the Backend defaults these describe the wrapper instead, which reads as an unaccelerated
    // priority-0 half whatever it holds, and [F64Context] composes all three from its halves.
    override val isPortable: Boolean get() = host?.isPortable ?: F64PlatformKernels.isPortable

    override val isAvailable: Boolean get() = host?.isAvailable ?: F64PlatformKernels.isAvailable

    override val priority: Int get() = host?.priority ?: F64PlatformKernels.priority

    /**
     * Crossing the host boundary costs more than the Level-1 work below this size. The Level1Benchmark
     * scalar/host runs at 63, 64, and 65 elements put the crossover at 64 for every routed primitive.
     * Keep these per-operation constants: a future measurement can move one without silently changing the
     * others.
     */
    private companion object {
        const val DOT_HOST_CROSSOVER = 64
        const val AXPY_HOST_CROSSOVER = 64
        const val SCALE_HOST_CROSSOVER = 64
        const val NRM2_HOST_CROSSOVER = 64
        const val ASUM_HOST_CROSSOVER = 64
        const val SWAP_HOST_CROSSOVER = 64
        const val ROT_HOST_CROSSOVER = 64
        const val ROTM_HOST_CROSSOVER = 64
    }

    private fun selected(len: Int, crossover: Int): F64Kernels =
        if (len < crossover) F64PlatformKernels else host ?: F64PlatformKernels

    // rotmg takes four scalars and has no len to threshold on, so it alone routes unconditionally: host when
    // present, compiled-in otherwise. Every routine with a length is gated, including the two rotations,
    // which used to reach the host at any length through this.
    private fun hostOrPlatform(): F64Kernels = host ?: F64PlatformKernels

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        selected(len, DOT_HOST_CROSSOVER).dot(a, aOff, b, bOff, len)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        selected(len, AXPY_HOST_CROSSOVER).axpy(y, yOff, alpha, x, xOff, len)
    }

    override fun axpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) =
        F64PlatformKernels.axpyArithmetic(y, yOff, alpha, x, xOff, len)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (alpha == 1.0) return
        selected(len, SCALE_HOST_CROSSOVER).scale(v, vOff, alpha, len)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double =
        selected(len, NRM2_HOST_CROSSOVER).nrm2(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double =
        selected(len, ASUM_HOST_CROSSOVER).asum(v, vOff, len)

    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens =
        hostOrPlatform().rotmg(d1, d2, x1, y1)

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
    ) = selected(len, ROTM_HOST_CROSSOVER).rotm(x, xOff, xStride, y, yOff, yStride, len, transformation)

    @Suppress("LongParameterList")
    override fun rot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) =
        selected(len, ROT_HOST_CROSSOVER).rot(x, xOff, y, yOff, len, c, s)

    /** Not routed, unlike every other routine here: only the compiled-in kernels fuse the four dots into
     *  one pass, and a host cannot, since its dot computes one at a time. */
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
    ) = F64PlatformKernels.dot4(a, aOff, stride, b, bOff, len, out, outOff)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) =
        selected(len, SWAP_HOST_CROSSOVER).swap(a, aOff, b, bOff, len)

    /** Not routed, as [dot4] is not: no host BLAS has a plain sum, so this always runs the compiled-in
     *  kernels. */
    override fun sum(v: DoubleArray, vOff: Int, len: Int): Double = F64PlatformKernels.sum(v, vOff, len)

    /** Not routed, as [dot4] is not: no host BLAS fuses a squared distance, so this always runs the
     *  compiled-in kernels. */
    override fun ssqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        F64PlatformKernels.ssqd(a, aOff, b, bOff, len)
}
