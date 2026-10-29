package com.eignex.koblas.vendor

import com.eignex.koblas.RouteKind


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
public class CallRoute public constructor(
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

