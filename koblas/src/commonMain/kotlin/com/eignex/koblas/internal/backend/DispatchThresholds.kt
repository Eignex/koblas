package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.dense.host.cblas.OpenBlasConfig

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

/** What this platform uses for double precision absent backend-specific configuration. */
internal expect val platformDispatchThresholds: DispatchThresholds

/** The dispatch policy for one OpenBLAS instance, with null configuration entries retaining platform defaults. */
internal fun openBlasDispatchThresholds(config: OpenBlasConfig): DispatchThresholds = DispatchThresholds(
    level1 = platformDispatchThresholds.level1,
    level2 = config.level2Min ?: platformDispatchThresholds.level2,
    level3 = config.level3Min ?: platformDispatchThresholds.level3,
    lapack = config.lapackMin ?: platformDispatchThresholds.lapack,
)

/** The compiled-in vector-kernel routing policy. */
internal val f64DispatchThresholds: DispatchThresholds get() = platformDispatchThresholds
