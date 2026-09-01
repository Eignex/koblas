package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.F64ModifiedGivens
import com.eignex.koblas.portableRotm
import com.eignex.koblas.portableRotmg

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
    public fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens = portableRotmg(d1, d2, x1, y1)

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
    ) {
        portableRotm(x, xOff, xStride, y, yOff, yStride, len, transformation)
    }

    /**
     * Exchange the two runs (BLAS `dswap`). Two loads and two stores an element, so an implementation is
     * bound by memory rather than by issue rate; the default is the plain loop for that reason.
     */
    public fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
        for (i in 0 until len) {
            val t = a[aOff + i]
            a[aOff + i] = b[bOff + i]
            b[bOff + i] = t
        }
    }

    /**
     * Four dots against a shared right operand. For r in 0..3, out(outOff + r) is the dot of the run at
     * aOff + r * stride with the run at bOff. Defaults to four [dot] calls, so an implementation that can
     * read the shared operand once for all four should override it.
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
    }

    private fun selected(len: Int, crossover: Int): F64Kernels =
        if (len < crossover) F64PlatformKernels else host ?: F64PlatformKernels

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
        selected.rotmg(d1, d2, x1, y1)

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
    ) = selected.rotm(x, xOff, xStride, y, yOff, yStride, len, transformation)

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
}
