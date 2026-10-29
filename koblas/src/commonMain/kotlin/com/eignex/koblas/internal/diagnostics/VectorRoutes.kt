package com.eignex.koblas.internal.diagnostics

import com.eignex.koblas.*

/** Shared dense and sparse classification, including empty and value-dependent calls. */
internal fun <O : Enum<O>> vectorRoute(
    operation: O,
    entryPoint: String,
    length: Int,
    selection: String,
    reached: String?,
    adapter: String? = null,
): VectorRoute<O> {
    require(length >= 0) { "negative operation length" }
    if (length == 0) {
        return VectorRoute(operation, RouteKind.NoWork, selection, entryPoint, adapter, "an operand has no entries")
    }
    return when {
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
}
