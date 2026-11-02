package com.eignex.koblas

import com.eignex.koblas.internal.configuration.CombinedEngine

/** Leaf providers in routing order, including their own capabilities and host provenance. */
public val KoblasEngine.providers: List<KoblasEngine> get() =
    if (this is CombinedEngine) components.toList() else listOf(this)

/** The provider bound by the dispatcher, before any value-dependent arithmetic fallback. */
internal fun KoblasEngine.providerFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int,
    contiguous: Boolean,
    borrowed: Boolean,
): String = if (operation.vector) {
    (if (this is CombinedEngine) vectors(operation, rows, contiguous) else this).id
} else {
    wholeCallFor(operation, rows, columns, depth, borrowed)?.id ?: "scheduled"
}

internal fun KoblasEngine.wholeCallFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int = 1,
    borrowed: Boolean = false,
): KoblasEngine? = when (this) {
    is CombinedEngine -> wholeCallFor(operation, rows, columns, depth, borrowed)
    else -> takeIf { operation in implementations }
}

internal fun KoblasEngine.implementationFor(
    operation: EngineImplementation,
    length: Int,
    contiguous: Boolean = true,
): String? = when {
    this is CombinedEngine -> implementationFor(operation, length, contiguous)
    operation !in implementations -> null
    dynamicAttribution(operation) -> null
    else -> name
}

internal fun KoblasEngine.implementationFor(
    work: PanelWork,
    rows: Int,
    columns: Int,
    contiguous: Boolean = true,
): String {
    val engine = if (this is CombinedEngine) panelImplementation else this
    return platformPanelImplementation(engine, work, rows, columns, contiguous) ?: engine.name
}

internal fun KoblasEngine.implementationsFor(rows: Int, columns: Int, depth: Int): List<String> {
    val engine = if (this is CombinedEngine) productImplementation else this
    return platformProductImplementations(engine, rows, columns, depth)
        ?: if (rows <= 0 || columns <= 0 || depth <= 0) emptyList() else listOf(engine.name)
}

internal fun KoblasEngine.implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> {
    val engine = if (this is CombinedEngine) triangularImplementation else this
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
