package com.eignex.koblas.internal.backend

/**
 * Defaults away from the JVM, where the portable kernels are scalar loops rather than SIMD, so levels 2,
 * 3 and LAPACK dispatch from the smallest size.
 */
internal actual val platformDispatchThresholds: DispatchThresholds =
    DispatchThresholds(
        level1 = NATIVE_LEVEL1_MIN,
        level2 = 0,
        level3 = 0,
        factorize = 0,
        symmetricFactorize = NATIVE_SYMMETRIC_FACTORIZE_MIN,
    )

/**
 * The stored-entry count from which a host library takes the symmetric sparse factorizations.
 *
 * The one factorization gate that is not zero here. The others are, because a foreign call costs little from
 * a native target and the portable kernels are scalar, so a library wins from the smallest sizes. These two
 * do not cross there: what the portable `L·Lᵀ` and `L·D·Lᵀ` lose to CHOLMOD is fill rather than call
 * overhead, and they take the ordering they are given. Measured on linuxX64 they win by three times at 214
 * stored entries and by a tenth at 387, and lose by a quarter at 484, so the crossing is near 417. See the
 * JVM threshold, which carries the rest of the reasoning and the same caveat about structure.
 */
private const val NATIVE_SYMMETRIC_FACTORIZE_MIN = 448

/** The run length from which a foreign call is worth making for a level-1 primitive. */
private const val NATIVE_LEVEL1_MIN = 32
