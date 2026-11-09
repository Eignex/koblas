package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/** The kernels a shared schedule run on this engine calls, which a built-in engine binds once. */
internal val KoblasEngine.panelProvider: KernelProvider get() = when (this) {
    is BuiltinEngine -> roles.panels
    is KernelProvider -> this
    else -> unsupported()
}

internal val KoblasEngine.productProvider: KernelProvider get() = when (this) {
    is BuiltinEngine -> roles.products
    is KernelProvider -> this
    else -> unsupported()
}

internal val KoblasEngine.triangularProvider: KernelProvider get() = when (this) {
    is BuiltinEngine -> roles.triangular
    is KernelProvider -> this
    else -> unsupported()
}

/** Selects the same primitive provider used by vector execution and alias preparation. */
internal fun KoblasEngine.vectorProviderFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): KoblasEngine = (this as? BuiltinEngine)?.vectorProvider(operation, length, contiguous) ?: this

/** The provider bound by the dispatcher, before any value-dependent arithmetic fallback. */
internal fun KoblasEngine.providerFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    contiguous: Boolean,
    borrowed: Boolean,
): String = if (operation.vector) {
    vectorProviderFor(operation, rows, contiguous).id
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
    val provider = (this as? BuiltinEngine)?.hostFor(operation, rows, columns, depth, borrowed) ?: this
    // Portable schedules retain the caller's selected kernels instead of a portable leaf's.
    return provider.takeIf { it !== ScalarEngine && (it !== this || operation in it.implementations) }
}

/**
 * The engine a whole call on this one is handed to, or null when this engine runs the shared schedule itself.
 *
 * Only a built-in engine hands calls to its selected host library, which must export the operation.
 */
internal fun KoblasEngine.routedWholeCall(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    borrowed: Boolean,
): KoblasEngine? = wholeCallFor(operation, rows, columns, depth, borrowed)?.takeIf { it !== this }

internal fun KoblasEngine.implementationFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): String? {
    val provider = vectorProviderFor(operation, length, contiguous)
    return provider.takeIf {
        operation in it.implementations && !(it is KernelProvider && it.dynamicAttribution(operation))
    }?.name
}
