package com.eignex.koblas

/**
 * JVM defaults, chosen from [mathBackend] because they hold only while the Vector API kernels are there.
 * The incubator module is opt-in, and without it the crossovers move down to the scalar ones.
 */
internal actual val platformDispatchThresholds: DispatchThresholds
    get() = if (mathBackend.startsWith("simd")) SIMD_THRESHOLDS else SCALAR_THRESHOLDS

/** For a JVM running the Vector API kernels. */
private val SIMD_THRESHOLDS = DispatchThresholds(
    level1 = Int.MAX_VALUE,
    level2 = Int.MAX_VALUE,
    level3 = JVM_LEVEL3_MIN,
    lapack = JVM_LAPACK_MIN,
)

/** For a JVM running the scalar loops, where the host library wins from the smallest sizes. */
private val SCALAR_THRESHOLDS = DispatchThresholds(level1 = 64, level2 = 0, level3 = 0, lapack = 0)

/** The matrix size from which the host BLAS takes the level-3 routines. */
private const val JVM_LEVEL3_MIN = 16

/** The matrix size from which the host LAPACK takes the factorizations. */
private const val JVM_LAPACK_MIN = 64

/** A system property first, then the environment, so a caller can use whichever suits their launcher. */
internal actual fun dispatchOverride(name: String): Int? = System.getProperty("koblas.dispatch.$name")?.toIntOrNull()
    ?: System.getenv("KOBLAS_DISPATCH_${name.uppercase()}")?.toIntOrNull()
