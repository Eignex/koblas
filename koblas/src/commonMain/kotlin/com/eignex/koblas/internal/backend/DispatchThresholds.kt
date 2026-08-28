package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.host.cblas.HostBlasConfig

/**
 * The smallest problem size at which dispatching to a native backend beats staying in Kotlin.
 * [Int.MAX_VALUE] keeps a family portable.
 *
 * Dense and kernel routing only, and every gate here is read as a dimension or a run length. The sparse
 * gates are [SparseDispatchThresholds], because their crossings are counted in stored entries: one number
 * cannot be both, and sharing one made a dense measurement move a sparse gate.
 *
 * @property level1 run length from which the level-1 primitives dispatch to a registered [F64Kernels].
 * @property level2 dimension from which the level-2 routines dispatch natively.
 * @property level3 dimension from which the level-3 routines dispatch natively.
 * @property factorize dimension from which the factorizations and the inverses built on them dispatch
 *   natively, the optional pivoted QR included.
 */
internal class DispatchThresholds(val level1: Int, val level2: Int, val level3: Int, val factorize: Int)

/**
 * The same for the sparse routines, counted in stored entries throughout, which is what sparse work scales
 * with: an order of a thousand with a diagonal and little else is smaller work than a dense hundred.
 *
 * One gate per library rather than one per routine. Each of these crosses against a different library with
 * its own fixed cost, so they are measured apart and a value here means nothing anywhere else.
 *
 * @property factorize stored entries from which the general sparse LU dispatches natively.
 * @property symmetric the same for the symmetric factorizations, which need no numerical pivoting and cross
 *   later than the general one.
 * @property qr the same for the sparse QR, which has its own crossing again.
 * @property product stored entries from which the sparse matrix products dispatch natively.
 */
internal class SparseDispatchThresholds(val factorize: Int, val symmetric: Int, val qr: Int, val product: Int)

/** What this platform uses for double precision absent backend-specific configuration. */
internal expect val platformDispatchThresholds: DispatchThresholds

/** The same for the sparse routines. Independent of the dense gates, and of whether SIMD kernels resolved. */
internal expect val platformSparseDispatchThresholds: SparseDispatchThresholds

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
): DispatchThresholds = DispatchThresholds(
    level1 = level1 ?: platformDispatchThresholds.level1,
    level2 = level2 ?: platformDispatchThresholds.level2,
    level3 = level3 ?: platformDispatchThresholds.level3,
    factorize = factorize ?: platformDispatchThresholds.factorize,
)

/** The sparse counterpart of [hostDispatchThresholds]; a null gate retains the platform default as there. */
internal fun hostSparseDispatchThresholds(
    factorize: Int? = null,
    symmetric: Int? = null,
    qr: Int? = null,
    product: Int? = null,
): SparseDispatchThresholds = SparseDispatchThresholds(
    factorize = factorize ?: platformSparseDispatchThresholds.factorize,
    symmetric = symmetric ?: platformSparseDispatchThresholds.symmetric,
    qr = qr ?: platformSparseDispatchThresholds.qr,
    product = product ?: platformSparseDispatchThresholds.product,
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
internal fun dispatchesLevel3(gate: Int, m: Int, n: Int, k: Int): Boolean =
    gate != Int.MAX_VALUE && m.toLong() * n * k >= gate.toLong() * gate * gate

/** The compiled-in vector-kernel routing policy. */
internal val f64DispatchThresholds: DispatchThresholds get() = platformDispatchThresholds
