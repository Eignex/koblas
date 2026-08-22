package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.installBackends

/**
 * JVM defaults, chosen from [simdAvailable] because they hold only while the Vector API kernels are there.
 * The incubator module is opt-in, and without it the crossovers move down to the scalar ones.
 *
 * Whether the module resolved is fixed for the process, which is what lets [f64DispatchThresholds] memoize
 * these. Reading the installed backend's name instead would be a value that moves under a memo that cannot:
 * an [installBackends] override would freeze whichever thresholds were in force at the first read.
 */
internal actual val platformDispatchThresholds: DispatchThresholds
    get() = if (simdAvailable) SIMD_THRESHOLDS else SCALAR_THRESHOLDS

/** For a JVM running the Vector API kernels. */
private val SIMD_THRESHOLDS = DispatchThresholds(
    level1 = Int.MAX_VALUE,
    level2 = Int.MAX_VALUE,
    level3 = JVM_LEVEL3_MIN,
    lapack = JVM_LAPACK_MIN,
)

/** For a JVM running the scalar loops, where the host library wins from the smallest sizes. */
private val SCALAR_THRESHOLDS =
    DispatchThresholds(level1 = JVM_SCALAR_LEVEL1_MIN, level2 = 0, level3 = 0, lapack = 0)

/** The run length from which a foreign call is worth making for a level-1 primitive. */
private const val JVM_SCALAR_LEVEL1_MIN = 64

/** The matrix size from which the host BLAS takes the level-3 routines. */
private const val JVM_LEVEL3_MIN = 16

/** The matrix size from which the host LAPACK takes the factorizations. */
private const val JVM_LAPACK_MIN = 64

/** A system property first, then the environment, so a caller can use whichever suits their launcher. */
internal actual fun dispatchOverride(element: ElementType, level: DispatchLevel): Int? =
    System.getProperty(ConfigurationKeys.dispatchProperty(element, level))?.toIntOrNull()
        ?: System.getenv(ConfigurationKeys.dispatchEnv(element, level))?.toIntOrNull()
