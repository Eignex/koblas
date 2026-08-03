package com.eignex.koblas

/**
 * Defaults away from the JVM, where the portable kernels are scalar loops rather than SIMD.
 *
 * Levels 2, 3 and LAPACK dispatch from the smallest size measured: on linuxX64 the host library won level
 * 2 by 1.6x to 15x even at n=16, and level 3 by 8x. Those thresholds exist so that can be retuned on a
 * platform where it does not hold, not because a crossover is known to be there.
 *
 * Level 1 is the exception, and the one place a crossover really was located. A foreign call costs roughly
 * 35ns on linuxX64, which is the entire cost of a short dot, so routing everything would make small vectors
 * slower. Measured against OpenBLAS, `axpy` breaks even at 32 and `dot` at 64; this is the higher of the
 * two, so neither regresses. The sweep is recorded in `Level1Benchmark`, and the value was only reachable
 * by recompiling until it joined this table.
 */
internal actual val platformDispatchThresholds: DispatchThresholds =
    DispatchThresholds(level1 = NATIVE_LEVEL1_MIN, level2 = 0, level3 = 0, lapack = 0)

private const val NATIVE_LEVEL1_MIN = 64

/** Environment only; there are no system properties outside the JVM. */
internal actual fun dispatchOverride(name: String): Int? = readEnv("KOBLAS_DISPATCH_${name.uppercase()}")?.toIntOrNull()

/** An environment variable, or null where the target has no environment to read. */
internal expect fun readEnv(name: String): String?
