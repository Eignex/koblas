package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.DispatchThresholds
import com.eignex.koblas.f64DispatchThresholds

/**
 * The vector-vector routines as a backend half, alongside [F64Blas] and [F64Lapack]. Implementations must
 * agree with [F64PlatformVectorKernels] to within rounding and read nothing outside the (offset, length)
 * window.
 *
 * To within rounding rather than exactly, because bit-for-bit is not a contract these routines can hold:
 * [F64PlatformVectorKernels] itself fuses its multiply-add above one lane width and does not below it, and
 * reduces over lanes as a tree rather than in order. Two conforming implementations can differ in the last
 * bits of a sum, and the reference routines are written not to depend on which one they got.
 *
 * A length of zero is legal everywhere and does nothing: the triangular and Householder kernels reach the
 * last row with an empty tail, so every routine here is called that way.
 */
public interface F64VectorKernels : Backend {
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
internal expect object F64PlatformVectorKernels : F64VectorKernels {
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
 * Picks between [F64PlatformVectorKernels] and a registered host backend by run length. The `alpha` guards
 * live here, so `axpy` by zero and `scale` by one are no-ops whichever kernel would have run.
 *
 * @property host a registered backend, used at or above [DispatchThresholds.level1]; null routes nothing.
 */
internal class F64RoutedVectorKernels(internal val host: F64VectorKernels?) : F64VectorKernels {
    override val name: String
        get() = if (host == null) F64PlatformVectorKernels.name else "${F64PlatformVectorKernels.name}+${host.name}"

    // Routing is all this class does, so what it reports about itself is what it routes to. Left to the
    // Backend defaults these describe the wrapper instead, which reads as an unaccelerated priority-0 half
    // whatever it holds, and [F64Context] composes all three from its halves.
    override val isPortable: Boolean get() = host?.isPortable ?: F64PlatformVectorKernels.isPortable

    override val isAvailable: Boolean get() = host?.isAvailable ?: F64PlatformVectorKernels.isAvailable

    override val priority: Int get() = host?.priority ?: F64PlatformVectorKernels.priority

    /** The kernels for a run of [len]: the host backend only when it exists and the run is long enough. */
    private fun forLength(len: Int): F64VectorKernels {
        val h = host
        return if (h != null && len >= f64DispatchThresholds.level1) h else F64PlatformVectorKernels
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
    ) = F64PlatformVectorKernels.dot4(a, aOff, stride, b, bOff, len, out, outOff)
}
