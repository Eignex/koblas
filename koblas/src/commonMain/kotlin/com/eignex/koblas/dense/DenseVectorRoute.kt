package com.eignex.koblas.dense

import com.eignex.koblas.vendor.RouteKind

/**
 * The implementation selected for a dense vector call, inspected before running it.
 *
 * [RouteKind.Delegated] identifies a portable fallback. [RouteKind.Composed] means the supplied facts
 * cannot identify one implementation, for example when a norm may retry through a rescaling loop.
 * Building a route allocates and belongs outside timed arithmetic.
 */
public class DenseVectorRoute internal constructor(
    /** The operation requested. */
    public val operation: DenseOperation,
    /** Whether the call is direct, delegated, composed or empty. */
    public val kind: RouteKind,
    /** The reached implementation, or the selected backend when the route is composed. */
    public val implementation: String,
    /** Why the call falls back or cannot identify one implementation. */
    public val reason: String?,
) {
    /** The vector operation within [implementation]. */
    public val entryPoint: String get() = operation.name.lowercase()

    /** Whether the call is an exact measurement of the selected implementation. */
    public val exactlyMeasurable: Boolean get() = kind == RouteKind.Direct

    override fun toString(): String = buildString {
        append(entryPoint).append(' ').append(kind.name.lowercase()).append(' ').append(implementation)
        reason?.let { append(" (").append(it).append(')') }
    }
}
