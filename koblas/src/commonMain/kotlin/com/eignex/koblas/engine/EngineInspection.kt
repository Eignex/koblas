package com.eignex.koblas.engine

import com.eignex.koblas.*
import com.eignex.koblas.internal.portable.PortableEngine

/** Leaf providers in routing order, including their own capabilities and host provenance. */
public val KoblasEngine.providers: List<KoblasEngine> get() =
    if (this is CombinedEngine) routing.backends.toList() else listOf(this)

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
    (if (this is CombinedEngine) routing.select(operation, rows, contiguous = contiguous) else this).id
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
    val provider = (this as? CombinedEngine)?.routing?.select(operation, rows, columns, depth, borrowed = borrowed)
        ?: this
    // Portable schedules retain the caller's selected kernels instead of a portable leaf's.
    return provider.takeIf { it !== PortableEngine && operation in it.implementations }
}

/** The engine a whole call on this one is handed to, or null when this engine runs the shared schedule itself. */
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
    val provider = (this as? CombinedEngine)?.routing?.select(operation, length, contiguous = contiguous) ?: this
    return provider.takeIf {
        operation in it.implementations && !(it is EngineKernels && it.dynamicAttribution(operation))
    }?.name
}

internal fun KoblasEngine.implementationFor(
    work: PanelWork,
    rows: Int,
    columns: Int,
    contiguous: Boolean = true,
): String {
    val engine = panelProvider
    return platformPanelImplementation(engine, work, rows, columns, contiguous) ?: engine.name
}

internal fun KoblasEngine.implementationsFor(rows: Int, columns: Int, depth: Int): List<String> {
    val engine = productProvider
    return platformProductImplementations(engine, rows, columns, depth)
        ?: if (rows <= 0 || columns <= 0 || depth <= 0) emptyList() else listOf(engine.name)
}

internal fun KoblasEngine.implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> {
    val engine = triangularProvider
    return platformTriangularImplementations(engine, order, sides, contiguous)
        ?: if (order <= 0 || sides <= 0) emptyList() else listOf(engine.name)
}

internal expect fun platformPanelImplementation(
    engine: KoblasEngine,
    work: PanelWork,
    rows: Int,
    columns: Int,
    contiguous: Boolean,
): String?
internal expect fun platformProductImplementations(
    engine: KoblasEngine,
    rows: Int,
    columns: Int,
    depth: Int,
): List<String>?
internal expect fun platformTriangularImplementations(
    engine: KoblasEngine,
    order: Int,
    sides: Int,
    contiguous: Boolean,
): List<String>?
