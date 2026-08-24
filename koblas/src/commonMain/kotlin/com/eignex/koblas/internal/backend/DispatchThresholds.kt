package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.dense.host.cblas.OpenBlasConfig

/**
 * The smallest problem size at which dispatching to a native backend beats staying in Kotlin.
 * [Int.MAX_VALUE] keeps a family portable.
 *
 * [level1] is honored by the routing in [F64VectorKernels]; the rest by the host backends, which share their
 * adapters across platforms and so read them everywhere. One gate per quantity a routine can be gated on
 * rather than one per routine: the factorizations and their inverses gate on a matrix dimension, and the
 * multi-column solves over their factors gate on a count of right-hand sides, which is not a dimension and
 * does not share a crossover with one.
 *
 * One value carries every gate an adapter reads, so a binding cannot wire one gate and forget another.
 *
 * @property level1 run length from which the level-1 primitives dispatch to a registered [F64VectorKernels].
 * @property level2 dimension from which the level-2 routines dispatch natively.
 * @property level3 dimension from which the level-3 routines dispatch natively.
 * @property factorize dimension from which the factorizations and the inverses built on them dispatch
 *   natively, the optional pivoted QR included.
 * @property factorizeRhs right-hand columns from which one blocked native solve over those factors beats
 *   a native call per column.
 */
internal class DispatchThresholds(
    val level1: Int,
    val level2: Int,
    val level3: Int,
    val factorize: Int,
    val factorizeRhs: Int = 1,
)

/** What this platform uses for double precision absent backend-specific configuration. */
internal expect val platformDispatchThresholds: DispatchThresholds

/**
 * The dispatch policy for one host backend. A null gate retains the platform default, so a host configures
 * only what it has measured.
 *
 * Every host binding resolves its gates through here, whichever library it binds, so the defaults are read
 * from one place and a new host has nothing to reimplement.
 */
internal fun hostDispatchThresholds(
    level1: Int? = null,
    level2: Int? = null,
    level3: Int? = null,
    factorize: Int? = null,
    factorizeRhs: Int? = null,
): DispatchThresholds = DispatchThresholds(
    level1 = level1 ?: platformDispatchThresholds.level1,
    level2 = level2 ?: platformDispatchThresholds.level2,
    level3 = level3 ?: platformDispatchThresholds.level3,
    factorize = factorize ?: platformDispatchThresholds.factorize,
    factorizeRhs = factorizeRhs ?: platformDispatchThresholds.factorizeRhs,
)

/** The dispatch policy for one OpenBLAS instance, every gate the configuration carries included. */
internal fun openBlasDispatchThresholds(config: OpenBlasConfig): DispatchThresholds = hostDispatchThresholds(
    level1 = config.level1Min,
    level2 = config.level2Min,
    level3 = config.level3Min,
    factorize = config.factorizeMin,
    factorizeRhs = config.factorizeRhsMin,
)

/** The compiled-in vector-kernel routing policy. */
internal val f64DispatchThresholds: DispatchThresholds get() = platformDispatchThresholds
