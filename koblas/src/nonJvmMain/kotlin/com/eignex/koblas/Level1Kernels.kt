package com.eignex.koblas

/**
 * Replacement level-1 kernels for the targets whose built-in ones are scalar.
 *
 * The [LinearAlgebra] seam covers level 2, level 3 and the factorizations, but `dot`, `axpy` and `scale`
 * sit below it: they are compiled per target and reached directly, so registering a backend does nothing
 * for them. On the JVM that is fine — those kernels are `jdk.incubator.vector` SIMD and beat a foreign
 * call at every size measured. On native they are plain loops, 4x to 7x slower than the JVM's, so a host
 * BLAS is worth reaching for once a run is long enough to cover the call.
 *
 * Only Linux and macOS consult this. The JS, Wasm, iOS and Windows targets call their scalar loops
 * directly, so they pay nothing for a seam they could never use.
 *
 * This is an implementation seam for backends, not an API for callers: use `dot`, `axpy` and `scale`,
 * which route through whatever is installed. It exists only on the non-JVM targets, since the JVM has
 * nothing to gain from it.
 *
 * Implementations must match the built-in kernels' contract exactly — same result for the same
 * `(offset, length)` window, no reads outside it — and must be safe to call from any thread that can
 * already call `dot`.
 */
interface Level1Kernels {
    /** `Sum a[aOff..aOff+len-1] * b[bOff..bOff+len-1]`, with `len >= 1`. */
    fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

    /** `y[yOff..yOff+len-1] += alpha * x[xOff..xOff+len-1]`, with `len >= 1`. */
    fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

    /** `v[vOff..vOff+len-1] *= alpha`, with `len >= 1`. */
    fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)
}

/**
 * The shortest run handed to installed [Level1Kernels]; anything shorter takes the scalar loop.
 *
 * A foreign call costs roughly 35ns on linuxX64, which is the entire cost of a short dot product, so
 * routing everything would make small vectors slower. Measured against OpenBLAS, `axpy` breaks even at
 * 32 and `dot` at 64; this is the higher of the two, so neither operation regresses. The full sweep is
 * recorded in `Primitives.hostBlas.kt`, and `Level1Benchmark` in koblas-bench reproduces it.
 */
internal const val HOST_LEVEL1_MIN_LENGTH = 64

@Suppress("ObjectPropertyNaming") // a mutable installed seam, not a constant
private var installedLevel1Kernels: Level1Kernels? = null

/**
 * Installs [kernels] for runs of at least [HOST_LEVEL1_MIN_LENGTH] elements, or clears them with `null`.
 *
 * koblas calls this itself at program start on Linux and macOS when the host provides OpenBLAS, next to
 * registering the [LinearAlgebra] backend. Above the threshold the measured win on linuxX64 runs from
 * 1.4x at length 64 to 10x at 1024; below it a foreign call costs more than the loop it replaces, which
 * is why routing is gated rather than unconditional.
 *
 * Not thread-safe against concurrent level-1 calls: install during startup, before other threads run.
 * Passing kernels that disagree with the built-in ones silently changes results everywhere in the
 * library, since `matVec`, the factorizations and the Cholesky updates all bottom out in these calls.
 */
fun installLevel1Kernels(kernels: Level1Kernels?) {
    installedLevel1Kernels = kernels
}

/** The installed [Level1Kernels], or `null` when the built-in loops are in use. */
internal fun installedLevel1Kernels(): Level1Kernels? = installedLevel1Kernels
