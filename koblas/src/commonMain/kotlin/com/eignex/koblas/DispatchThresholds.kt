package com.eignex.koblas

import com.eignex.koblas.dense.VectorKernels

/**
 * The smallest problem size at which dispatching to a native backend beats staying in Kotlin.
 *
 * One threshold per BLAS level rather than per routine, because the crossover is a property of the level:
 * it follows the ratio of work to data. Level 1 is a run length rather than a dimension, and is the one
 * that used to be a compile-time constant with no override — which meant retuning it for a machine with a
 * different vector width took a recompile. Level 2 does `O(n²)` work over `O(n²)` data, so there is nothing
 * to amortize a foreign call against and the answer barely depends on which routine it is; level 3 and the
 * factorizations do `O(n³)` work over the same data and cross over once the cube outgrows the call.
 *
 * A routine may deviate when a measurement says so, but it has to carry the numbers in its KDoc — twelve
 * independent constants would be unmaintainable, three plus an earned exception is not.
 *
 * The defaults differ per platform and for a good reason: the JVM's portable kernels are Vector API SIMD
 * and beat a foreign call at level 2 outright, while on Kotlin/Native the same kernels are scalar loops
 * and lose to it from the smallest sizes measured.
 *
 * A threshold larger than any problem a caller will pose keeps a level portable; [Int.MAX_VALUE] is the
 * blunt way to say that, and it needs no name of its own.
 *
 * @property level1 run length from which the level-1 primitives dispatch to a registered [VectorKernels].
 * @property level2 dimension from which the level-2 routines dispatch natively.
 * @property level3 dimension from which the level-3 routines dispatch natively.
 * @property lapack dimension from which the factorizations dispatch natively.
 */
internal class DispatchThresholds(val level1: Int, val level2: Int, val level3: Int, val lapack: Int)

/** What this platform uses absent an override; see [DispatchThresholds]. */
internal expect val platformDispatchThresholds: DispatchThresholds

/**
 * Reads an override for `koblas.dispatch.<name>`, or null when unset.
 *
 * Resolved once at startup rather than per call: these are read on the dispatch path, and an unusual
 * machine — a different OpenBLAS build, a wider vector unit — is a reason to retune without recompiling,
 * not a reason to pay for a lookup on every call.
 */
internal expect fun dispatchOverride(name: String): Int?

/** The thresholds in force, overrides applied. */
internal val dispatchThresholds: DispatchThresholds by lazy {
    val defaults = platformDispatchThresholds
    DispatchThresholds(
        level1 = dispatchOverride("level1") ?: defaults.level1,
        level2 = dispatchOverride("level2") ?: defaults.level2,
        level3 = dispatchOverride("level3") ?: defaults.level3,
        lapack = dispatchOverride("lapack") ?: defaults.lapack,
    )
}
