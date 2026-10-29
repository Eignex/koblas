package com.eignex.koblas

import com.eignex.koblas.RouteKind

/**
 * The route for a dense or sparse vector operation, inspected before running it.
 *
 * A delegated route names the fallback. A composed route names the selected backend because the operand
 * values decide which implementation finishes the call, and an empty one performs no arithmetic for either
 * to be measured by. Route construction belongs outside timed work.
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
