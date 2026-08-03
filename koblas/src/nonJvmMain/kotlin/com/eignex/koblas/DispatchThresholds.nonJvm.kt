package com.eignex.koblas

/**
 * Defaults away from the JVM, where the portable kernels are scalar loops rather than SIMD.
 *
 * Every level dispatches from the smallest size measured: on linuxX64 the host library won level 2 by 1.6x
 * to 15x even at n=16, and level 3 by 8x. The thresholds exist so that can be retuned on a platform where
 * it does not hold, not because a crossover is known to be there.
 */
internal actual val platformDispatchThresholds: DispatchThresholds =
    DispatchThresholds(level2 = 0, level3 = 0, lapack = 0)

/** Environment only; there are no system properties outside the JVM. */
internal actual fun dispatchOverride(name: String): Int? = readEnv("KOBLAS_DISPATCH_${name.uppercase()}")?.toIntOrNull()

/** An environment variable, or null where the target has no environment to read. */
internal expect fun readEnv(name: String): String?
