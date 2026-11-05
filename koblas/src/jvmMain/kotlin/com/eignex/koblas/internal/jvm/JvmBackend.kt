package com.eignex.koblas.internal.jvm

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.engine.BuiltinEngines
import com.eignex.koblas.engine.Vendor
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.koblas

/** Portable entry points replaced by the multi-release JAR on desktop JDK 25 or later. */
internal object JvmBackend {
    val contributions: List<EngineKernels>? get() = null

    val default: KoblasEngine get() = BuiltinEngines.scalar

    @Suppress("UNUSED_PARAMETER", "FunctionOnlyReturningConstant") // Matches the desktop selection entry point.
    fun open(vendor: Vendor): KoblasEngine? = null
}
