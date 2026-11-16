package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.internal.host.HostBlas
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/** The kernels a shared schedule run on this engine calls, which a built-in engine binds once. */
internal val KernelProvider.panelProvider: KernelProvider get() = when (this) {
    is BuiltinEngine -> roles.panels
    else -> this
}

internal val KernelProvider.productProvider: KernelProvider get() = when (this) {
    is BuiltinEngine -> roles.products
    else -> this
}

internal val KernelProvider.triangularProvider: KernelProvider get() = when (this) {
    is BuiltinEngine -> roles.triangular
    else -> this
}

/** Selects the same primitive provider used by vector execution and alias preparation. */
internal fun KernelProvider.vectorProviderFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): KernelProvider = (this as? BuiltinEngine)?.vectorProvider(operation, length, contiguous) ?: this

/** The provider bound by the dispatcher, before any value-dependent arithmetic fallback. */
internal fun KernelProvider.providerFor(
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
internal fun KernelProvider.wholeCallFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int = 1,
    borrowed: Boolean = false,
): KernelProvider? {
    val provider = (this as? BuiltinEngine)?.hostFor(operation, rows, columns, depth, borrowed) ?: this
    // Portable schedules retain the caller's selected kernels instead of a portable leaf's.
    return provider.takeIf { it !== ScalarEngine && (it !== this || operation in it.implementations) }
}

/**
 * The engine a whole call on this one is handed to, or null when this engine runs the shared schedule itself.
 *
 * Only a built-in engine hands calls to its selected host library, which must export the operation.
 */
internal fun KernelProvider.routedWholeCall(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    borrowed: Boolean,
): KernelProvider? = wholeCallFor(operation, rows, columns, depth, borrowed)?.takeIf { it !== this }

internal fun KernelProvider.implementationFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): String? {
    val provider = vectorProviderFor(operation, length, contiguous)
    return provider.takeIf {
        operation in it.implementations && !it.dynamicAttribution(operation)
    }?.name
}

internal val KoblasEngine.panelProvider: KernelProvider get() = kernels.panelProvider
internal val KoblasEngine.productProvider: KernelProvider get() = kernels.productProvider
internal val KoblasEngine.triangularProvider: KernelProvider get() = kernels.triangularProvider

internal fun KoblasEngine.vectorProviderFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): KernelProvider = kernels.vectorProviderFor(operation, length, contiguous)

internal fun KoblasEngine.wholeCallFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int = 1,
    borrowed: Boolean = false,
): KernelProvider? = kernels.wholeCallFor(operation, rows, columns, depth, borrowed)

internal fun KoblasEngine.routedWholeCall(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    borrowed: Boolean,
): KernelProvider? = kernels.routedWholeCall(operation, rows, columns, depth, borrowed)

internal fun KoblasEngine.providerFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    contiguous: Boolean,
    borrowed: Boolean,
): String = kernels.providerFor(operation, rows, columns, depth, contiguous, borrowed)

internal fun KoblasEngine.implementationFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): String? = kernels.implementationFor(operation, length, contiguous)

/** Resolve availability before preparation can write or borrow storage. */
internal fun KoblasEngine.hostForCall(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    borrowed: Boolean,
): HostBlas? = kernels.hostForCall(operation, rows, columns, depth, borrowed)

internal fun KernelProvider.hostForCall(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    borrowed: Boolean,
): HostBlas? {
    val backend = this
    if (backend is HostBlas) {
        if (operation !in backend.implementations) {
            throw UnsupportedOperationException("${operation.entryPoint} is not exported by ${backend.name}")
        }
        return backend
    }
    return (backend as? BuiltinEngine)?.hostFor(operation, rows, columns, depth, borrowed) as? HostBlas
}
