package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.DispatchThresholds
import com.eignex.koblas.dispatchThresholds

/**
 * The vector-vector routines — `dot`, `axpy`, `scale`, `nrm2`, `asum` — as a backend half, alongside [Blas]
 * and [Lapack].
 *
 * Every implementation of these routines is one of these, which was not always true and is the point of the
 * interface. koblas ships a compile-time specialization per target ([PlatformVectorKernels]: Vector API SIMD
 * on the JVM, scalar loops elsewhere) and may additionally have a host BLAS registered; both are
 * `VectorKernels`, and [RoutedVectorKernels] picks between them by run length.
 *
 * ### Why these sit below the [Blas] seam
 *
 * They do nanoseconds of work, so a virtual call per invocation would cost more than the kernel. That is why
 * the compiled-in kernels exist at all, and why a registered backend is consulted only once a run is long
 * enough for a foreign call to pay for itself — [DispatchThresholds.level1]. What it is *not* is a reason for
 * the compiled kernels to have a different shape from a registered one: they used to be loose
 * `expect fun platformDot`-style functions implementing no interface, so "the kernels" meant two unrelated
 * things depending on where you looked, and the seam needed `null` to mean "the other kind". One interface
 * removes both problems.
 *
 * ### What is not here, and why
 *
 * `iamax` is excluded on a contract question rather than a performance one. koblas documents its
 * tie-breaking — first by index for a dense vector, storage order for a sparse one — and its behaviour for
 * an all-unstored vector. `idamax` implementations do not agree with each other on where a `NaN` ranks, so
 * routing it would make the answer depend on whether a host library happened to be installed. It could
 * join once that behaviour has been checked against the libraries koblas actually dispatches to.
 *
 * `copy` and `swap` are excluded on the arithmetic: `copyInto` and an element exchange are already the
 * best available, so a foreign call could only lose.
 *
 * Implementations must agree with [PlatformVectorKernels] exactly — same result for the same
 * `(offset, length)` window, no reads outside it — and must be safe to call from any thread that can
 * already call `dot`. Kernels that disagree silently change results everywhere in the library, since
 * `gemv`, the factorizations and the eta updates all bottom out here.
 */
interface VectorKernels : Backend {
    /** `Sum a[aOff..aOff+len-1] * b[bOff..bOff+len-1]`, with `len >= 1`. */
    fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

    /** `y[yOff..yOff+len-1] += alpha * x[xOff..xOff+len-1]`, with `len >= 1`. */
    fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

    /** `v[vOff..vOff+len-1] *= alpha`, with `len >= 1`. */
    fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)

    /**
     * Euclidean norm of `v[vOff..vOff+len-1]`, with `len >= 1` (BLAS `dnrm2`).
     *
     * Must be accurate for components whose squares overflow or underflow — koblas's built-in kernel
     * rescales to stay in range, and netlib `dnrm2` does the same, so a plain `sqrt(sum of squares)` is
     * not a valid implementation.
     */
    fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double

    /** `Sum |v[vOff..vOff+len-1]|`, with `len >= 1` (BLAS `dasum`). */
    fun asum(v: DoubleArray, vOff: Int, len: Int): Double

    /**
     * Four dots against a shared right operand: `out[outOff + r] = Sum a[aOff + r*stride + i] * b[bOff + i]`
     * for `r` in `0..3`. Four columns of a column-major matrix against one vector, which is the shape
     * `gemv` and the `Aᵀ·B` `gemm` branch need.
     *
     * The one routine here with a default, because it is the one with no BLAS counterpart: `ddot` exists,
     * batched `ddot` does not. The default is four [dot] calls, which is exactly what a host backend should
     * do — so a host implements this by inheriting it, rather than by not having it.
     *
     * An implementation that can do better should: the point of the shape is that each `b` segment is loaded
     * once for all four columns and the four accumulator chains are independent, which cuts load traffic and
     * breaks the dependency chain a single [dot] is limited by. [PlatformVectorKernels] overrides it for
     * that reason.
     */
    @Suppress("LongParameterList") // four column offsets plus the shared operand
    fun dot4(
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
 * The kernels compiled into this target: `jdk.incubator.vector` SIMD on the JVM (falling back to scalar
 * loops when the module is absent), scalar loops everywhere else.
 *
 * An `expect object` rather than an `expect val` of interface type, deliberately. Being a known singleton
 * keeps a below-threshold call a direct call that the compiler can inline; behind an interface-typed
 * property it would become a virtual dispatch on every `dot`, which the JVM might devirtualize but the
 * native and web targets would simply pay. Those are the targets that reach a host BLAS, so they are the
 * ones that can least afford it.
 *
 * Always present, so there is always something to call and no null to interpret. Its [Backend.name] is what
 * `mathBackend` reports.
 */
internal expect object PlatformVectorKernels : VectorKernels {
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
 * Picks between [PlatformVectorKernels] and a registered host backend by run length.
 *
 * A `VectorKernels` itself, so it composes where any other does and nothing downstream needs to know a
 * routing decision is happening. With no [host] every call forwards to the compiled-in kernels, which is
 * the case on every target that cannot reach a host BLAS.
 *
 * The `alpha` guards live here rather than at the call sites that used to carry them: `axpy` by zero and
 * `scale` by one are no-ops, and skipping them before the length comparison keeps that true whichever
 * kernel would have run.
 *
 * @property host a registered backend, used at or above [DispatchThresholds.level1]; null routes
 *   nothing. Readable so a test can re-register it after clearing the registry.
 */
internal class RoutedVectorKernels(internal val host: VectorKernels?) : VectorKernels {
    override val name: String
        get() = if (host == null) PlatformVectorKernels.name else "${PlatformVectorKernels.name}+${host.name}"

    /** The kernels to use for a run of [len]: the host backend only when it exists and the run is long
     *  enough for a foreign call to pay for itself. */
    private fun forLength(len: Int): VectorKernels {
        val h = host
        return if (h != null && len >= dispatchThresholds.level1) h else PlatformVectorKernels
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

    /**
     * Not routed, unlike every other routine here: always the compiled-in kernels.
     *
     * A host backend inherits the interface default, which is four independent `ddot` calls — and that
     * discards the entire reason this routine exists, since the shared `b` loads and the four independent
     * accumulator chains are the win. One vectorized pass beating four foreign calls is the expected
     * outcome, not a measured one, so this preserves the behaviour koblas has always had rather than
     * changing it on a guess. Worth measuring alongside the other thresholds.
     */
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
    ) = PlatformVectorKernels.dot4(a, aOff, stride, b, bOff, len, out, outOff)
}
