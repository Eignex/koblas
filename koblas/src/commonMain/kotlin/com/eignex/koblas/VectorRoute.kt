package com.eignex.koblas

import com.eignex.koblas.vendor.RouteKind

/**
 * The route for a dense or sparse vector operation, inspected before running it.
 *
 * A delegated route names the fallback. A composed route names the selected backend because the operand
 * values decide which implementation finishes the call. Route construction belongs outside timed work.
 */
public class VectorRoute<O : Enum<O>> internal constructor(
    /** The operation requested. */
    public val operation: O,
    /** Whether the selected implementation runs directly, delegates or composes the result. */
    public val kind: RouteKind,
    /** The reached implementation, or the selected backend for a composed route. */
    public val implementation: String,
    /** The operation within [implementation]. */
    public val entryPoint: String,
    /** Operand adaptation or the delegated implementation, when applicable. */
    public val adapter: String?,
    /** Why the call delegates or cannot identify one implementation. */
    public val reason: String?,
) {
    /** Whether this is an exact measurement of the selected implementation. */
    public val exactlyMeasurable: Boolean get() = kind == RouteKind.Direct

    override fun toString(): String = buildString {
        append(operation.name.lowercase()).append(' ').append(kind.name.lowercase())
        append(' ').append(implementation).append(' ').append(entryPoint)
        adapter?.let { append(" via ").append(it) }
        reason?.let { append(" (").append(it).append(')') }
    }
}

/** Classifies the same selected and reached implementations for both vector families. */
internal fun <O : Enum<O>> vectorRoute(
    operation: O,
    entryPoint: String,
    selection: String,
    reached: String?,
    adapter: String? = null,
): VectorRoute<O> = when {
    reached == null -> VectorRoute(
        operation,
        RouteKind.Composed,
        selection,
        entryPoint,
        adapter,
        "$selection decides $entryPoint based on the values",
    )

    reached == selection -> VectorRoute(operation, RouteKind.Direct, reached, entryPoint, adapter, null)

    else -> VectorRoute(
        operation,
        RouteKind.Delegated,
        reached,
        entryPoint,
        reached,
        "$selection has no $entryPoint kernel for this call",
    )
}
