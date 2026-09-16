package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector

/** How a call reached its result. */
public enum class RouteKind {
    /** The vendor entry point ran over these operands. What it cost to reach it is named by [CallRoute.adapter]. */
    Direct,

    /** The whole call was handed to another implementation, which [CallRoute.adapter] names. */
    Delegated,

    /** The result was assembled from more than one call, or from a vendor call plus arithmetic here. */
    Composed,

    /** The operation's own contract says there is nothing to do, so nothing ran. */
    NoWork,

    /** Nothing ran and nothing can; [CallRoute.reason] says why. */
    Unsupported,
}

/**
 * What a concrete call does, derived from the same decision the call itself makes.
 *
 * A route is evidence about one call with one set of arguments, not about an operation in general. The same
 * [BlasOperation] can be [RouteKind.Direct] where the library exports it and [RouteKind.Composed] where it
 * does not, and a benchmark that treats those as one measurement is reporting the composition's time as the
 * entry point's.
 *
 * Building one allocates, so it happens when a caller asks and never inside a timed call. [Blas.routeOf]
 * and the execution path read the same internal plan, so a route cannot drift from what runs.
 */
public class CallRoute internal constructor(
    /** The operation requested. */
    public val operation: BlasOperation,
    /** How the call was served. */
    public val kind: RouteKind,
    /** The vendor that ran it, or null when nothing vendor-supplied did. */
    public val vendor: Vendor?,
    /** The resolved symbol, or null when no vendor entry point was called. */
    public val entryPoint: String?,
    /** How the operands reached the library, or what composed the result, or null when neither applies. */
    public val adapter: String?,
    /** Why the call was [RouteKind.Unsupported] or [RouteKind.NoWork], or null otherwise. */
    public val reason: String?,
) {
    /**
     * Whether this call is admissible as an exact measurement of [vendor] running [operation].
     *
     * A transfer into native memory still qualifies, because it is part of what a caller pays to reach the
     * vendor at all. A composed or delegated call does not: its time belongs to the composition, not to the
     * entry point it is being reported under.
     */
    public val exactlyMeasurable: Boolean get() = kind == RouteKind.Direct && vendor != null

    override fun toString(): String = buildString {
        append(operation.name.lowercase())
        append(' ')
        append(kind.name.lowercase())
        vendor?.let { append(" ").append(it.vendorName) }
        entryPoint?.let { append(" ").append(it) }
        adapter?.let { append(" via ").append(it) }
        reason?.let { append(" (").append(it).append(')') }
    }
}

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
 * route against [Blas.directlyImplemented] means both have to agree before a timing is admitted.
 *
 * The platform transfer does not disqualify a call. On the JVM every operand is copied into native memory to
 * reach the library at all, and the route says so; the arm is still measuring that vendor's work.
 */
public fun exactArmRejection(
    blas: Blas,
    operation: BlasOperation,
    matrices: List<DenseMatrix> = emptyList(),
    vectors: List<DenseVector> = emptyList(),
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
