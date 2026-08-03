package com.eignex.koblas

/**
 * JVM defaults, which depend on which primitive kernels this runtime actually got.
 *
 * Every JVM number behind these thresholds was measured with the Vector API kernels active, and they only
 * hold while those kernels are there. The incubator module is opt-in: a consumer who launches without
 * `--add-modules=jdk.incubator.vector` runs the scalar loops instead, which are 4x to 7x slower, and the
 * crossovers move down accordingly — keeping `level2` at [Int.MAX_VALUE] in that case would pin them to
 * scalar arithmetic where a foreign call wins comfortably.
 *
 * So the choice is made from [mathBackend], the same seam that decides which kernels ran.
 */
internal actual val platformDispatchThresholds: DispatchThresholds
    get() = if (mathBackend.startsWith("simd")) SIMD_THRESHOLDS else SCALAR_THRESHOLDS

/** Measured on this platform with the Vector API kernels; see [JVM_LEVEL3_MIN] and [JVM_LAPACK_MIN]. */
private val SIMD_THRESHOLDS = DispatchThresholds(
    level2 = Int.MAX_VALUE,
    level3 = JVM_LEVEL3_MIN,
    lapack = JVM_LAPACK_MIN,
)

/**
 * For a JVM running the scalar loops, where the portable side is the same Kotlin the native targets compile
 * and the host library wins from the smallest sizes.
 *
 * Inherited from the linuxX64 measurements rather than measured on a scalar JVM: there, against scalar
 * kernels, the host BLAS won level 2 by 1.6x to 15x from n=16 and level 3 by 8x to 20x. The JIT is not LLVM,
 * so the numbers will differ in detail, but not in which side wins — and the alternative, applying
 * SIMD-derived thresholds to a runtime that has no SIMD, is wrong by a much wider margin.
 */
private val SCALAR_THRESHOLDS = DispatchThresholds(level2 = 0, level3 = 0, lapack = 0)

private const val JVM_LEVEL3_MIN = 32

/**
 * The factorizations cross over between 32 and 64 (`luFactorAndSolve`, us/op): SIMD wins 1.83x at 16 and
 * 1.59x at 32, then native takes 1.02x at 64, 1.41x at 128 and 1.73x at 256.
 *
 * `qr`, `ldl` and `rcond` are covered by analogy rather than measurement — they are the same `O(n^3)` shape
 * over the same data, which is the premise of having one threshold per level. If one of them turns out to
 * behave differently, it earns its own constant and its own numbers.
 *
 * Read alongside the delegated routines in the same sweep: `luSolve` and `ldlSolveInto` run identical code
 * on both sides and still differed by 1.02x to 1.20x, so anything inside that band is noise, not signal.
 */
private const val JVM_LAPACK_MIN = 64

/** A system property first, then the environment, so a caller can use whichever suits their launcher. */
internal actual fun dispatchOverride(name: String): Int? = System.getProperty("koblas.dispatch.$name")?.toIntOrNull()
    ?: System.getenv("KOBLAS_DISPATCH_${name.uppercase()}")?.toIntOrNull()
