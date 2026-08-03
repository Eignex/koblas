package com.eignex.koblas

/**
 * JVM defaults.
 *
 * Level 2 is [NEVER_NATIVE]: the Vector API kernels beat a foreign call at every size measured, by 3x to
 * 15x, because `O(n²)` work cannot amortize the call. The level-3 and factorization values are the
 * measured crossovers against the host OpenBLAS; `Level3Benchmark` and `SolveBenchmark` reproduce them.
 */
internal actual val platformDispatchThresholds: DispatchThresholds =
    DispatchThresholds(level2 = NEVER_NATIVE, level3 = JVM_LEVEL3_MIN, lapack = JVM_LAPACK_MIN)

/**
 * Level 3 crosses over between n=4 and n=64, depending on the routine (us/op, native against SIMD):
 *
 *     n            4      16      32      64     128      256
 *     gemm     0.252   0.535   2.206  13.001  83.606  552.960
 *     simd     0.055   0.900   4.949  35.773 263.871 2171.467
 *     syrk     0.369   0.739   2.424  10.780  82.338  502.246
 *     simd     0.035   0.407   2.000  13.331 115.968 1147.993
 *
 * The routines inside the level disagree: `gemm` is ahead from 16, `syrk` only from 64, and `syrk` with a
 * single triangle later still. 32 is the value that minimizes total absolute regret across the sweep
 * (3.0us/call against 3.2 at 16 and 4.1 at 64), which is the honest way to pick one number for a level
 * whose members differ. At n=4 the foreign call costs 4x to 10x the arithmetic, so any threshold at or
 * above 16 covers the case that matters most.
 */
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
