package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector

/**
 * The route of one call, derived from the operand addressing that the call itself acts on.
 *
 * Both the execution path and [VendorBlas.routeOf] end here, which is the point: a benchmark that asks what a
 * call does gets the answer from the decision the call makes, not from a second description that can drift
 * away from it. An engine name or a requested arm is not evidence of anything on its own.
 *
 * [transfer] is the platform's own storage adaptation, which is not optional and not free. On the JVM every
 * operand is copied into native memory for the downcall, so the honest route says so and a measurement that
 * includes the call includes that copy. On Kotlin/Native the array is pinned instead and there is nothing to
 * name.
 */
internal fun routeFor(
    operation: VendorOperation,
    vendor: Vendor,
    exported: Boolean,
    matrices: List<DenseMatrix>,
    vectors: List<DenseVector>,
    transfer: String?,
): CallRoute {
    noWorkReason(matrices, vectors)?.let { return noWorkRoute(operation, vendor, it) }
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
    val adapter = transfer
    return CallRoute(
        operation = operation,
        kind = RouteKind.Direct,
        vendor = vendor,
        entryPoint = operation.entryPoint,
        adapter = adapter,
        reason = null,
    )
}

/**
 * Why this call has nothing to do, or null when it has.
 *
 * The one rule both the execution paths and [VendorBlas.routeOf] read, so a call that returns without reaching
 * BLAS cannot be described as having reached it. Every operation here exits early on an empty operand, and that
 * exit is what this expresses; no scalar makes a call no-work, because none of the bound entry points is
 * skipped on a zero multiplier and inventing that would describe a call that does run as one that does not.
 */
internal fun noWorkReason(matrices: List<DenseMatrix>, vectors: List<DenseVector>): String? {
    if (vectors.any { it.size == 0 }) return "an operand has no entries"
    if (matrices.any { it.rows == 0 || it.cols == 0 }) return "an operand has no entries"
    return null
}

/** The route of a call whose own contract says there is nothing to do. */
internal fun noWorkRoute(operation: VendorOperation, vendor: Vendor, reason: String): CallRoute = CallRoute(
    operation = operation,
    kind = RouteKind.NoWork,
    vendor = vendor,
    entryPoint = null,
    adapter = null,
    reason = reason,
)

/** How an operation is assembled when the selected vendor does not export it. */
private fun compositionOf(operation: VendorOperation): String = when (operation) {
    VendorOperation.Gemmt -> "${VendorOperation.Gemm.entryPoint} plus triangle copy"
    else -> "composed here"
}
