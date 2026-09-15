package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixWindow

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
    matrices: List<MatrixWindow>,
    transfer: String?,
): CallRoute {
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
    val staged = effectiveAddressing(operation, matrices).any { it == Addressing.Staged }
    val adapter = listOfNotNull(transfer, if (staged) "packed copy" else null)
        .takeIf { it.isNotEmpty() }
        ?.joinToString(" plus ")
    return CallRoute(
        operation = operation,
        kind = RouteKind.Direct,
        vendor = vendor,
        entryPoint = operation.entryPoint,
        adapter = adapter,
        reason = null,
    )
}

/** How an operation is assembled when the selected vendor does not export it. */
private fun compositionOf(operation: VendorOperation): String = when (operation) {
    VendorOperation.Gemmt -> "${VendorOperation.Gemm.entryPoint} plus triangle copy"
    else -> "composed here"
}
