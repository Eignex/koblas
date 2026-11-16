package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.internal.host.HostBlas
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/** The kernels a shared schedule run on this engine calls, which a built-in engine binds once. */
internal val KernelProvider.panelProvider: PanelKernels get() = when (this) {
    is BuiltinEngine -> roles.panels
    else -> this as? PanelKernels ?: missingRole()
}

internal val KernelProvider.productProvider: PackedProductKernels get() = when (this) {
    is BuiltinEngine -> roles.products
    else -> this as? PackedProductKernels ?: missingRole()
}

internal val KernelProvider.triangularProvider: TriangularKernels get() = when (this) {
    is BuiltinEngine -> roles.triangular
    else -> this as? TriangularKernels ?: missingRole()
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
    val provider = if (this is BuiltinEngine) hostFor(operation, rows, columns, depth, borrowed) else this
    return provider?.takeIf { operation in it.implementations && it !== ScalarEngine }
}

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

internal val KoblasEngine.panelProvider: PanelKernels get() = kernels.panelProvider
internal val KoblasEngine.productProvider: PackedProductKernels get() = kernels.productProvider
internal val KoblasEngine.triangularProvider: TriangularKernels get() = kernels.triangularProvider

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
internal fun KernelProvider.hostForCall(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    borrowed: Boolean,
): HostBlas? {
    val selected = wholeCallFor(operation, rows, columns, depth, borrowed)
    if (selected is HostBlas) return selected
    if (this is HostBlas) {
        throw UnsupportedOperationException("${operation.entryPoint} is not exported by $name")
    }
    return null
}
