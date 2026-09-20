package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector

/**
 * The route of one call, derived from the operand addressing that the call itself acts on.
 *
 * Both the execution path and [Blas.routeOf] end here, which is the point: a benchmark that asks what a
 * call does gets the answer from the decision the call makes, not from a second description that can drift
 * away from it. An engine name or a requested arm is not evidence of anything on its own.
 *
 * [transfer] is the platform's own storage adaptation, which is not optional and not free. On the JVM every
 * operand is copied into native memory for the downcall, so the honest route says so and a measurement that
 * includes the call includes that copy. On Kotlin/Native the array is pinned instead and there is nothing to
 * name.
 */
internal fun routeFor(
    operation: BlasOperation,
    vendor: Vendor,
    exported: Boolean,
    matrices: List<DenseMatrix>,
    vectors: List<DenseVector>,
    transfer: String?,
): CallRoute {
    // Level 3 still scales a nonempty destination when the product depth is zero. Its destination is the
    // last matrix operand, and it is the only extent the bindings use for their quick return.
    val workMatrices = if (operation.level == 3) matrices.takeLast(1) else matrices
    noWorkReason(workMatrices, vectors)?.let { return noWorkRoute(operation, vendor, it) }
    if (!exported) {
        return CallRoute(
            operation = operation,
            kind = RouteKind.Composed,
            vendor = vendor,
            entryPoint = null,
            adapter = compositionOf(operation),
            reason = "${vendor.vendorName} does not export ${operation.entryPoint}",
        )
    }
    // Every operand is contiguous column-major, so the platform transfer is the only adaptation there is.
    return CallRoute(
        operation = operation,
        kind = RouteKind.Direct,
        vendor = vendor,
        entryPoint = operation.entryPoint,
        adapter = transfer,
        reason = null,
    )
}

/**
 * Whether [x] leaves a call with nothing to do.
 *
 * The rule itself, which an execution path asks of the one operand governing its quick return and
 * [noWorkReason] asks of every operand a route was given. Stating it once is what keeps a call that returns
 * without reaching BLAS from being described as having reached it, and asking it of an operand rather than
 * of a list of them is what keeps the question off the arithmetic path's allocation budget.
 */
internal fun noWork(x: DenseVector): Boolean = x.size == 0

/** The same rule over a matrix operand. */
internal fun noWork(a: DenseMatrix): Boolean = a.rows == 0 || a.cols == 0

/**
 * Why this call has nothing to do, or null when it has.
 *
 * Callers pass the operands whose extents govern their quick return: Level 3 passes only its destination
 * because a zero product depth still scales it. No scalar makes a call no-work, because none of the bound
 * entry points is skipped on a zero multiplier and inventing that would describe a call that does run as one
 * that does not.
 */
internal fun noWorkReason(matrices: List<DenseMatrix>, vectors: List<DenseVector>): String? {
    for (i in vectors.indices) if (noWork(vectors[i])) return NO_ENTRIES
    for (i in matrices.indices) if (noWork(matrices[i])) return NO_ENTRIES
    return null
}

/** What a call with an empty operand reports, which is the same for either operand shape. */
private const val NO_ENTRIES = "an operand has no entries"

/** The route of a call whose own contract says there is nothing to do. */
internal fun noWorkRoute(operation: BlasOperation, vendor: Vendor, reason: String): CallRoute = CallRoute(
    operation = operation,
    kind = RouteKind.NoWork,
    vendor = vendor,
    entryPoint = null,
    adapter = null,
    reason = reason,
)

/** How an operation is assembled when the selected vendor does not export it. */
private fun compositionOf(operation: BlasOperation): String = when (operation) {
    BlasOperation.Gemmt -> "${BlasOperation.Gemm.entryPoint} plus triangle copy"
    else -> "composed here"
}
