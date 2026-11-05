package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/** The kernels a shared schedule run on this engine calls, which a composed engine binds once. */
internal val KoblasEngine.panelProvider: EngineKernels get() = when (this) {
    is CombinedEngine -> panelImplementation
    is EngineKernels -> this
    else -> unsupported()
}

internal val KoblasEngine.productProvider: EngineKernels get() = when (this) {
    is CombinedEngine -> productImplementation
    is EngineKernels -> this
    else -> unsupported()
}

internal val KoblasEngine.triangularProvider: EngineKernels get() = when (this) {
    is CombinedEngine -> triangularImplementation
    is EngineKernels -> this
    else -> unsupported()
}

/** The provider bound by the dispatcher, before any value-dependent arithmetic fallback. */
internal fun KoblasEngine.providerFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    contiguous: Boolean,
    borrowed: Boolean,
): String = if (operation.vector) {
    (if (this is CombinedEngine) vectorProvider(operation, rows, contiguous) else this).id
} else {
    wholeCallFor(operation, rows, columns, depth, borrowed)?.id ?: "scheduled"
}

/** The engine a whole call of these extents goes to, or null when the shared schedule runs it. */
internal fun KoblasEngine.wholeCallFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int = 1,
    borrowed: Boolean = false,
): KoblasEngine? {
    val provider = (this as? CombinedEngine)?.routing?.select(operation, rows, columns, depth, borrowed) ?: this
    // Portable schedules retain the caller's selected kernels instead of a portable leaf's.
    return provider.takeIf { it !== ScalarEngine && operation in it.implementations }
}

/**
 * The engine a whole call on this one is handed to, or null when this engine runs the shared schedule itself.
 *
 * Only a composition hands calls over, and every leaf it routes to supplies the operation and is not itself.
 */
internal fun KoblasEngine.routedWholeCall(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    borrowed: Boolean,
): KoblasEngine? = (this as? CombinedEngine)?.routing?.select(operation, rows, columns, depth, borrowed)
    ?.takeIf { it !== ScalarEngine }

internal fun KoblasEngine.implementationFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): String? {
    val provider = (this as? CombinedEngine)?.vectorProvider(operation, length, contiguous) ?: this
    return provider.takeIf {
        operation in it.implementations && !(it is EngineKernels && it.dynamicAttribution(operation))
    }?.name
}
