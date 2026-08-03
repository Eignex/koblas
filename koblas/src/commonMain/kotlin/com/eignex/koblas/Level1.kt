package com.eignex.koblas

import kotlin.concurrent.Volatile

/**
 * The level-1 kernels — `dot`, `axpy`, `scale` — as a registered backend half, alongside [Blas] and
 * [Lapack].
 *
 * These sit below the [Blas] seam rather than on it, because they do nanoseconds of work and a virtual
 * call per invocation would cost more than the kernel. koblas therefore ships them as compile-time
 * specializations (Vector API SIMD on the JVM, scalar loops elsewhere) and consults this seam only once a
 * run is long enough that a foreign call pays for itself — [DispatchThresholds.level1].
 *
 * ### Why this is not part of [LinearAlgebra]
 *
 * [Blas] and [Lapack] compose into [LinearAlgebra] because a backend can supply both and the portable
 * reference supplies a default for each. Level 1 cannot join them: the default *is* the compiled-in
 * primitive, not an object, and [ReferenceLinearAlgebra] deliberately does not implement this interface.
 * If it did, `dot` would resolve to a reference implementation that itself calls `dot`, which is a
 * recursion rather than a fallback. So the three halves share the registration verb, the priority
 * ranking, the threshold table and the diagnostics, while level 1 keeps a `null`-means-built-in default.
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
 * Implementations must match the built-in kernels' contract exactly — same result for the same
 * `(offset, length)` window, no reads outside it — and must be safe to call from any thread that can
 * already call `dot`. Passing kernels that disagree with the built-in ones silently changes results
 * everywhere in the library, since `gemv`, the factorizations and the eta updates all bottom out here.
 */
interface Level1 : Backend {
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
}

/**
 * The level-1 backend the primitives route to, or null when the built-in kernels are in use.
 *
 * A plain field rather than a function or a lazy value, because every `dot` in the library reads it. The
 * primitives test it *before* comparing lengths: on a target where nothing is ever registered — the JVM,
 * iOS, Wasm — that makes the whole seam one always-null field read, cheaper than resolving a threshold
 * would be. See [denseDot] for the ordering.
 */
@Volatile
internal var activeLevel1: Level1? = null
    private set

@Volatile
private var installedLevel1: Level1? = null

@Volatile
private var registeredLevel1: Level1? = null

private fun recomposeLevel1() {
    activeLevel1 = installedLevel1 ?: registeredLevel1
}

/**
 * Overrides which [Level1] the primitives use, taking precedence over [registerLevel1]. Passing null
 * restores automatic selection, and clearing both restores the built-in kernels.
 *
 * The counterpart of [installLinearAlgebra] for this half. Not thread-safe against concurrent level-1
 * calls: install during startup, before other threads run.
 */
fun installLevel1(backend: Level1?) {
    installedLevel1 = backend
    recomposeLevel1()
}

/**
 * Offers [backend] for automatic selection, ranked by [Backend.priority] against other registered level-1
 * backends. This is how koblas activates the host kernels on the targets that can reach a host BLAS; see
 * [registerBlas] for the same mechanism on the other halves.
 */
fun registerLevel1(backend: Level1) {
    val current = registeredLevel1
    if (current == null || backend.priority > current.priority) {
        registeredLevel1 = backend
        recomposeLevel1()
    }
}

/** Test hook: clears level-1 registration so selection tests are order-independent. */
internal fun resetRegisteredLevel1() {
    installedLevel1 = null
    registeredLevel1 = null
    recomposeLevel1()
}
