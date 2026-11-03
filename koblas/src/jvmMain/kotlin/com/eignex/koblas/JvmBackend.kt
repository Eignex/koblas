package com.eignex.koblas

import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.Vendor

/** Portable entry points replaced by the multi-release JAR on desktop JDK 25 or later. */
internal object JvmBackend {
    val contributions: List<EngineKernels>? get() = null

    val default: KoblasEngine get() = BuiltinEngines.scalar

    @Suppress("UNUSED_PARAMETER", "FunctionOnlyReturningConstant") // Matches the desktop selection entry point.
    fun open(vendor: Vendor): KoblasEngine? = null

    @Suppress("UNUSED_PARAMETER", "FunctionOnlyReturningConstant")
    fun panelImplementation(
        engine: KoblasEngine,
        work: PanelWork,
        rows: Int,
        columns: Int,
        contiguous: Boolean,
    ): String? = null

    @Suppress("UNUSED_PARAMETER", "FunctionOnlyReturningConstant")
    fun productImplementations(engine: KoblasEngine, rows: Int, columns: Int, depth: Int): List<String>? = null

    @Suppress("UNUSED_PARAMETER", "FunctionOnlyReturningConstant")
    fun triangularImplementations(engine: KoblasEngine, order: Int, sides: Int, contiguous: Boolean): List<String>? =
        null
}
