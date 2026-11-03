package com.eignex.koblas.engine

import com.eignex.koblas.*

@Suppress("UNUSED_PARAMETER") // Native providers execute only their declared body.
internal actual fun platformPanelImplementation(
    engine: KoblasEngine,
    work: PanelWork,
    rows: Int,
    columns: Int,
    contiguous: Boolean,
): String? = null

@Suppress("UNUSED_PARAMETER") // Native providers execute only their declared body.
internal actual fun platformProductImplementations(
    engine: KoblasEngine,
    rows: Int,
    columns: Int,
    depth: Int,
): List<String>? = null

@Suppress("UNUSED_PARAMETER") // Native providers execute only their declared body.
internal actual fun platformTriangularImplementations(
    engine: KoblasEngine,
    order: Int,
    sides: Int,
    contiguous: Boolean,
): List<String>? = null
