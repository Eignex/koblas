package com.eignex.koblas.internal.backend

/**
 * Defaults away from the JVM, where the portable kernels are scalar loops rather than SIMD, so levels 2,
 * 3 and LAPACK dispatch from the smallest size.
 */
internal actual val platformDispatchThresholds: DispatchThresholds =
    DispatchThresholds(level1 = NATIVE_LEVEL1_MIN, level2 = 0, level3 = 0, lapack = 0)

/** The run length from which a foreign call is worth making for a level-1 primitive. */
private const val NATIVE_LEVEL1_MIN = 64
