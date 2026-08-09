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
    level1 = Int.MAX_VALUE,
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
private val SCALAR_THRESHOLDS = DispatchThresholds(level1 = 64, level2 = 0, level3 = 0, lapack = 0)

/**
 * Native from 16 up, on all three level-3 routines (us/op, native against SIMD). `gemm`: 0.30 vs 0.24 at
 * n=8, 0.55 vs 1.07 at 16, 2.33 vs 5.88 at 32, 655 vs 2324 at 256. `syrk` full: 0.40 vs 0.29 at 8, 0.74 vs
 * 1.32 at 16, 496 vs 2589 at 256. `syrk` lower: 0.31 vs 0.29 at 8, 0.50 vs 1.28 at 16, 366 vs 2509 at 256.
 *
 * All three agree on where the crossover is, which is the premise of one threshold per level rather than per
 * routine. Eight stays portable; below that a foreign call is most of the cost.
 */
private const val JVM_LEVEL3_MIN = 16

/**
 * The factorizations cross over at 64 (`luFactorAndSolve`, us/op, native against SIMD): 0.60 vs 0.33 at n=8,
 * 1.66 vs 1.14 at 16, 6.61 vs 4.50 at 32, then native takes it at 20.5 vs 22.3 at 64, 79 vs 141 at 128 and
 * 401 vs 946 at 256.
 *
 * `qr`, `ldl` and `rcond` are covered by analogy rather than measurement — the same `O(n³)` shape over the
 * same data, which is the premise of one threshold per level. If one behaves differently it earns its own
 * constant and its own numbers.
 */
private const val JVM_LAPACK_MIN = 64

/** A system property first, then the environment, so a caller can use whichever suits their launcher. */
internal actual fun dispatchOverride(name: String): Int? = System.getProperty("koblas.dispatch.$name")?.toIntOrNull()
    ?: System.getenv("KOBLAS_DISPATCH_${name.uppercase()}")?.toIntOrNull()
