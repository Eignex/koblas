package com.eignex.koblas

import com.eignex.koblas.internal.configuration.CombinedEngine

/** Positive contributions grouped without exposing the private pieces of a composition. */
public val KoblasEngine.capabilities: Map<String, Set<EngineImplementation>> get() = components.groupBy { it.id }
    .mapValues { (_, parts) -> parts.flatMapTo(mutableSetOf()) { it.implementations } }

internal val KoblasEngine.components: List<KoblasEngine> get() =
    if (this is CombinedEngine) components else listOf(this)

internal fun KoblasEngine.wholeCallFor(
    operation: EngineImplementation,
    rows: Int,
    columns: Int,
    depth: Int = 1,
    borrowed: Boolean = false,
): KoblasEngine? = when (this) {
    is CombinedEngine -> wholeCallFor(operation, rows, columns, depth, borrowed)

    PortableEngine -> null

    else -> if (operation in implementations) {
        this
    } else {
        throw UnsupportedOperationException(
            "$operation is not supplied by $name",
        )
    }
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
    return if (engine is CombinedEngine) {
        engine.implementationFor(work, rows, columns, contiguous)
    } else {
        platformPanelImplementation(engine, work, rows, columns, contiguous) ?: engine.name
    }
}

internal fun KoblasEngine.implementationsFor(rows: Int, columns: Int, depth: Int): List<String> {
    val engine = if (this is CombinedEngine) productImplementation else this
    return if (engine is CombinedEngine) {
        engine.implementationsFor(rows, columns, depth)
    } else {
        platformProductImplementations(engine, rows, columns, depth)
            ?: if (rows <= 0 || columns <= 0 || depth <= 0) emptyList() else listOf(engine.name)
    }
}

internal fun KoblasEngine.implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> {
    val engine = if (this is CombinedEngine) triangularImplementation else this
    return if (engine is CombinedEngine) {
        engine.implementationsFor(order, sides, contiguous)
    } else {
        platformTriangularImplementations(engine, order, sides, contiguous)
            ?: if (order <= 0 || sides <= 0) emptyList() else listOf(engine.name)
    }
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
