package com.eignex.koblas

import com.eignex.koblas.dense.VectorKernels

/**
 * The smallest problem size at which dispatching to a native backend beats staying in Kotlin. One threshold
 * per BLAS level, and [Int.MAX_VALUE] keeps a level portable.
 *
 * [level1] is honored on every platform, by the routing in [VectorKernels]. The other three are read only by
 * the JVM host backend; the native CBLAS backend dispatches whatever the size, so setting them there has no
 * effect. Its defaults are 0, so the two agree today, and only an override would tell them apart.
 *
 * @property level1 run length from which the level-1 primitives dispatch to a registered [VectorKernels].
 * @property level2 dimension from which the level-2 routines dispatch natively. JVM only.
 * @property level3 dimension from which the level-3 routines dispatch natively. JVM only.
 * @property lapack dimension from which the factorizations dispatch natively. JVM only.
 */
internal class DispatchThresholds(val level1: Int, val level2: Int, val level3: Int, val lapack: Int)

/** What this platform uses absent an override; see [DispatchThresholds]. */
internal expect val platformDispatchThresholds: DispatchThresholds

/** Reads an override for `koblas.dispatch.<name>`, or null when unset. */
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
