package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.internal.backend.f64DispatchThresholds

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
    /** The run length from which this backend replaces the compiled-in kernels, or null for the platform default. */
    public val minDispatchLength: Int? get() = null

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
     * Adds `mult * a(k)` into y and returns the dot of the same run with x, in one pass. A symmetric
     * product needs both halves of every column, and one pass reads the column once rather than twice.
     */
    @Suppress("LongParameterList") // three runs and the multiplier, all of which the caller holds
    public fun symvColumn(
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: Double,
        len: Int,
    ): Double {
        var s = 0.0
        for (i in 0 until len) {
            val ai = a[aOff + i]
            s += ai * x[xOff + i]
            y[yOff + i] += mult * ai
        }
        return s
    }

    /**
     * Four [symvColumn] runs at `aOff + r * stride`, their dots landing in `out(r)`. Defaults to four
     * calls, so an implementation that can read x and y once for all four should override it.
     */
    @Suppress("LongParameterList") // four column offsets over three runs, plus the multipliers
    public fun symvColumn4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: DoubleArray,
        out: DoubleArray,
        len: Int,
    ) {
        for (r in 0 until 4) out[r] = symvColumn(a, aOff + r * stride, x, xOff, y, yOff, mult[r], len)
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

/**
 * The kernels compiled into this target, `jdk.incubator.vector` SIMD on the JVM and scalar loops
 * elsewhere. Its [Backend.name] is what `mathBackend` reports.
 */
internal expect object F64PlatformKernels : F64Kernels {
    override val name: String

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double

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
 * Picks between [F64PlatformKernels] and a registered host backend by run length. The `alpha` guards
 * live here, so `axpy` by zero and `scale` by one are no-ops whichever kernel would have run.
 *
 * @property host a registered backend, used at or above the configured minimum; null routes nothing.
 */
internal class F64RoutedKernels(internal val host: F64Kernels?) : F64Kernels {
    private val minLength: Int = host?.minDispatchLength ?: f64DispatchThresholds.level1
    override val name: String
        get() = if (host == null) F64PlatformKernels.name else "${F64PlatformKernels.name}+${host.name}"

    // Routing is all this class does, so what it reports about itself is what it routes to. Left to the
    // Backend defaults these describe the wrapper instead, which reads as an unaccelerated priority-0 half
    // whatever it holds, and [F64Context] composes all three from its halves.
    override val isPortable: Boolean get() = host?.isPortable ?: F64PlatformKernels.isPortable

    override val isAvailable: Boolean get() = host?.isAvailable ?: F64PlatformKernels.isAvailable

    override val priority: Int get() = host?.priority ?: F64PlatformKernels.priority

    /** The kernels for a run of [len]: the host backend only when it exists and the run is long enough. */
    private fun forLength(len: Int): F64Kernels {
        val h = host
        return if (h != null && len >= minLength) h else F64PlatformKernels
    }

    override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
        forLength(len).dot(a, aOff, b, bOff, len)

    override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        forLength(len).axpy(y, yOff, alpha, x, xOff, len)
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (alpha == 1.0) return
        forLength(len).scale(v, vOff, alpha, len)
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = forLength(len).nrm2(v, vOff, len)

    override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = forLength(len).asum(v, vOff, len)

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
        forLength(len).swap(a, aOff, b, bOff, len)

    @Suppress("LongParameterList") // three runs and the multiplier, matching the seam
    override fun symvColumn(
        a: DoubleArray,
        aOff: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: Double,
        len: Int,
    ): Double = F64PlatformKernels.symvColumn(a, aOff, x, xOff, y, yOff, mult, len)

    @Suppress("LongParameterList") // four column offsets over three runs, matching the seam
    override fun symvColumn4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        mult: DoubleArray,
        out: DoubleArray,
        len: Int,
    ) = F64PlatformKernels.symvColumn4(a, aOff, stride, x, xOff, y, yOff, mult, out, len)
}
