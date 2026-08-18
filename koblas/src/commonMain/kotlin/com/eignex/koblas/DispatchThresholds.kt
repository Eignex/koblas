package com.eignex.koblas

import com.eignex.koblas.dense.F64VectorKernels

/**
 * The smallest problem size at which dispatching to a native backend beats staying in Kotlin. One threshold
 * per BLAS level, and [Int.MAX_VALUE] keeps a level portable.
 *
 * [level1] is honored by the routing in [F64VectorKernels]; the rest by the host BLAS backends, which share
 * one adapter and so read them on every platform. The native defaults are 0, so native dispatches at any
 * size unless an override says otherwise.
 *
 * @property level1 run length from which the level-1 primitives dispatch to a registered [F64VectorKernels].
 * @property level2 dimension from which the level-2 routines dispatch natively.
 * @property level3 dimension from which the level-3 routines dispatch natively.
 * @property lapack dimension from which the factorizations dispatch natively.
 */
internal class DispatchThresholds(val level1: Int, val level2: Int, val level3: Int, val lapack: Int)

/** One [DispatchThresholds] entry, the granularity at which a caller can override the routing. */
internal enum class DispatchLevel {
    LEVEL1,
    LEVEL2,
    LEVEL3,
    LAPACK,
    ;

    /** How this level is written in [ConfigurationKeys.DISPATCH_PROPERTY_PREFIX]. */
    val key: String get() = name.lowercase()
}

/** What this platform uses for double precision absent an override; see [DispatchThresholds]. */
internal expect val platformDispatchThresholds: DispatchThresholds

/** Reads the override for [element]'s [level], or null when unset. */
internal expect fun dispatchOverride(element: ElementType, level: DispatchLevel): Int?

/**
 * The thresholds in force for [element], its [defaults] with any override applied. Crossovers are counted
 * in elements rather than bytes, so a narrower element type measures its own and passes them here.
 */
internal fun dispatchThresholdsFor(element: ElementType, defaults: DispatchThresholds): DispatchThresholds =
    DispatchThresholds(
        level1 = dispatchOverride(element, DispatchLevel.LEVEL1) ?: defaults.level1,
        level2 = dispatchOverride(element, DispatchLevel.LEVEL2) ?: defaults.level2,
        level3 = dispatchOverride(element, DispatchLevel.LEVEL3) ?: defaults.level3,
        lapack = dispatchOverride(element, DispatchLevel.LAPACK) ?: defaults.lapack,
    )

/** The double-precision thresholds in force, overrides applied. */
internal val f64DispatchThresholds: DispatchThresholds by lazy {
    dispatchThresholdsFor(ElementType.F64, platformDispatchThresholds)
}
