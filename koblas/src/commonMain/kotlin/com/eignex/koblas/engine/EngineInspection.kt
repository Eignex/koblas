package com.eignex.koblas.engine

import com.eignex.koblas.*
import com.eignex.koblas.internal.portable.PortableEngine

/** Leaf providers in routing order, including their own capabilities and host provenance. */
public val KoblasEngine.providers: List<KoblasEngine> get() =
    arithmeticEngine.let { if (it is CombinedEngine) it.routing.backends.toList() else listOf(it) }

internal val KoblasEngine.panelProvider: KoblasEngine get() =
    arithmeticEngine.let { if (it is CombinedEngine) it.panelImplementation else it }

internal val KoblasEngine.productProvider: KoblasEngine get() =
    arithmeticEngine.let { if (it is CombinedEngine) it.productImplementation else it }

internal val KoblasEngine.triangularProvider: KoblasEngine get() =
    arithmeticEngine.let { if (it is CombinedEngine) it.triangularImplementation else it }

/** The provider bound by the dispatcher, before any value-dependent arithmetic fallback. */
internal fun KoblasEngine.providerFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    contiguous: Boolean,
    borrowed: Boolean,
): String = if (operation.vector) {
    arithmeticEngine.let {
        (if (it is CombinedEngine) it.routing.select(operation, rows, contiguous = contiguous) else it).id
    }
} else {
    wholeCallFor(operation, rows, columns, depth, borrowed)?.id ?: "scheduled"
}

internal fun KoblasEngine.wholeCallFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int = 1,
    borrowed: Boolean = false,
): KoblasEngine? {
    val provider = (arithmeticEngine as? CombinedEngine)?.routing
        ?.select(operation, rows, columns, depth, borrowed = borrowed) ?: arithmeticEngine
    // Portable schedules retain the caller's selected kernels and workspace without allocating vector views.
    return provider.takeIf { it !== PortableEngine && operation in it.implementations }
}

internal fun KoblasEngine.implementationFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): String? {
    val provider = (arithmeticEngine as? CombinedEngine)?.routing?.select(
        operation,
        length,
        contiguous = contiguous,
    ) ?: arithmeticEngine
    return provider.takeIf { operation in it.implementations && !it.dynamicAttribution(operation) }?.name
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
