package com.eignex.koblas.vendor

/** How a call reached its result. */
public enum class RouteKind {
    /** The vendor entry point ran over these operands. Staging, if any, is named by [CallRoute.adapter]. */
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
 * [VendorOperation] can be [RouteKind.Direct] for a column-major panel and [RouteKind.Direct] with a staging
 * adapter for a window whose strides BLAS cannot express, and a benchmark that treats those as one measurement
 * is reporting the staging cost as vendor arithmetic.
 *
 * Building one allocates, so it happens when a caller asks and never inside a timed call. [VendorBlas.routeOf]
 * and the execution path read the same internal plan, so a route cannot drift from what runs.
 */
public class CallRoute internal constructor(
    /** The operation requested. */
    public val operation: VendorOperation,
    /** How the call was served. */
    public val kind: RouteKind,
    /** The vendor that ran it, or null when nothing vendor-supplied did. */
    public val vendor: Vendor?,
    /** The resolved symbol, or null when no vendor entry point was called. */
    public val entryPoint: String?,
    /** Staging, delegation, or composition detail, or null when the operands went straight through. */
    public val adapter: String?,
    /** Why the call was [RouteKind.Unsupported] or [RouteKind.NoWork], or null otherwise. */
    public val reason: String?,
) {
    /**
     * Whether this call is admissible as an exact measurement of [vendor] running [operation].
     *
     * A staged call still qualifies, because staging is part of what a caller pays to reach the vendor. A
     * composed or delegated one does not: its time belongs to the composition, not to the entry point.
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
