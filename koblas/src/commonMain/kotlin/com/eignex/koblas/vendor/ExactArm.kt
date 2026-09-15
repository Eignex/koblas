package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow

/**
 * Why a call may not be timed as an exact measurement of [blas] running [operation], or null when it may.
 *
 * An exact comparison arm exists to answer one question: how fast is this implementation at this operation.
 * A call that reached the answer some other way does not answer it, however it was labelled. So the checks
 * here are deliberately the ones a plausible mislabelling would fail: a route that is not direct, a vendor or
 * entry point that is not the one asked for, and an operation the layer does not implement at all.
 *
 * The last is not redundant. A layer can produce a direct-looking route for an operation it only composes, and
 * that is exactly the shape of a benchmark that reports composition as vendor arithmetic. Cross-checking the
 * route against [VendorBlas.directlyImplemented] means both have to agree before a timing is admitted.
 *
 * Staging does not disqualify a call. A caller reaching the vendor through a window BLAS cannot address pays
 * the copy as part of reaching it, and the route says so; the arm is still measuring that vendor's work.
 */
public fun exactArmRejection(
    blas: VendorBlas,
    operation: VendorOperation,
    matrices: List<MatrixWindow> = emptyList(),
    vectors: List<VectorWindow> = emptyList(),
): String? {
    if (operation !in blas.directlyImplemented) {
        return "${blas.vendor.vendorName} does not implement ${operation.entryPoint} directly"
    }
    val route = blas.routeOf(operation, matrices, vectors)
    if (route.kind != RouteKind.Direct) {
        return "route is ${route.kind.name.lowercase()}, not a direct call: ${route.reason ?: route}"
    }
    if (route.vendor != blas.vendor) {
        return "route names ${route.vendor?.vendorName ?: "no vendor"}, not ${blas.vendor.vendorName}"
    }
    if (route.entryPoint != operation.entryPoint) {
        return "route resolves ${route.entryPoint ?: "no entry point"}, not ${operation.entryPoint}"
    }
    return null
}
