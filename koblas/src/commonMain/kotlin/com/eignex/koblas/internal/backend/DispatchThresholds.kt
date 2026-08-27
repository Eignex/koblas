package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.host.cblas.HostBlasConfig

/**
 * The smallest problem size at which dispatching to a native backend beats staying in Kotlin.
 * [Int.MAX_VALUE] keeps a family portable.
 *
 * [level1] is honored by the routing in [F64Kernels]; the rest by the host backends, which share their
 * adapters across platforms and so read them everywhere. One gate per quantity a routine can be gated on
 * rather than one per routine: the factorizations and their inverses gate on a matrix dimension, and the
 * multi-column solves over their factors gate on a count of right-hand sides, which is not a dimension and
 * does not share a crossover with one.
 *
 * One value carries every gate an adapter reads, so a binding cannot wire one gate and forget another.
 *
 * @property level1 run length from which the level-1 primitives dispatch to a registered [F64Kernels].
 * @property level2 dimension from which the level-2 routines dispatch natively.
 * @property level3 dimension from which the level-3 routines dispatch natively.
 * @property factorize size from which the factorizations and the inverses built on them dispatch natively,
 *   the optional pivoted QR included. A dense routine reads it as a dimension and a sparse one as a count of
 *   stored entries, since that is what each one's work scales with; one number, measured against whichever
 *   quantity pays for the crossing.
 * @property symmetricFactorize the same, for the factorizations that need no numerical pivoting. They cross
 *   later than the general one, so they gate on a number of their own rather than dispatching early on its.
 * @property sparseProduct stored entries from which the sparse matrix products dispatch natively. Separate
 *   from [level2], which a vector-kernel platform sets beyond reach because its own kernels win the dense
 *   level-2 routines; there are no vector kernels for a sparse product, so that reasoning does not carry.
 * @property sparseQr stored entries from which sparse QR dispatches to SPQR. It has its own crossing rather
 *   than inheriting the general sparse-LU factorization gate.
 */
internal class DispatchThresholds(
    val level1: Int,
    val level2: Int,
    val level3: Int,
    val factorize: Int,
    val symmetricFactorize: Int = factorize,
    val sparseProduct: Int = level2,
    val sparseQr: Int = factorize,
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
    symmetricFactorize: Int? = null,
    sparseProduct: Int? = null,
    sparseQr: Int? = null,
): DispatchThresholds = DispatchThresholds(
    level1 = level1 ?: platformDispatchThresholds.level1,
    level2 = level2 ?: platformDispatchThresholds.level2,
    level3 = level3 ?: platformDispatchThresholds.level3,
    factorize = factorize ?: platformDispatchThresholds.factorize,
    symmetricFactorize = symmetricFactorize ?: platformDispatchThresholds.symmetricFactorize,
    sparseProduct = sparseProduct ?: platformDispatchThresholds.sparseProduct,
    sparseQr = sparseQr ?: platformDispatchThresholds.sparseQr,
)

/** The dispatch policy for one OpenBLAS instance, every gate the configuration carries included. */
internal fun hostBlasDispatchThresholds(config: HostBlasConfig): DispatchThresholds = hostDispatchThresholds(
    level1 = config.level1Min,
    level2 = config.level2Min,
    level3 = config.level3Min,
    factorize = config.factorizeMin,
)

/**
 * Whether a level-3 product whose work is `m · n · k` is worth handing to a native library, against a [gate]
 * measured as the dimension at which a cube of that side crosses over.
 *
 * The work at that crossover is the cube of the gate, and a shape reaching the same work has paid for the
 * crossing however narrow its narrowest side is. Reading the gate as a dimension instead leaves a wide
 * matrix times a two-column one portable on the two, whatever the other two dimensions are.
 *
 * Never less permissive than the dimension it was measured as, since every side at or above [gate] is a
 * work volume at or above the cube; [Int.MAX_VALUE] keeps the family portable as it does everywhere else.
 */
internal fun dispatchesLevel3(gate: Int, m: Int, n: Int, k: Int): Boolean {
    if (gate == Int.MAX_VALUE) return false
    return m.toLong() * n * k >= gate.toLong() * gate * gate
}

/** The compiled-in vector-kernel routing policy. */
internal val f64DispatchThresholds: DispatchThresholds get() = platformDispatchThresholds
