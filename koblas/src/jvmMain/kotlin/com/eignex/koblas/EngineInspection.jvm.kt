package com.eignex.koblas

internal actual fun platformPanelImplementation(
    engine: KoblasEngine,
    work: PanelWork,
    rows: Int,
    columns: Int,
    contiguous: Boolean,
): String? = JvmBackend.panelImplementation(
    engine,
    work,
    rows,
    columns,
    contiguous,
)

internal actual fun platformProductImplementations(
    engine: KoblasEngine,
    rows: Int,
    columns: Int,
    depth: Int,
): List<String>? = JvmBackend.productImplementations(
    engine,
    rows,
    columns,
    depth,
)

internal actual fun platformTriangularImplementations(
    engine: KoblasEngine,
    order: Int,
    sides: Int,
    contiguous: Boolean,
): List<String>? = JvmBackend.triangularImplementations(
    engine,
    order,
    sides,
    contiguous,
)
