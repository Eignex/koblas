package com.eignex.koblas

import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.Vendor

/** Portable entry points replaced by the multi-release JAR on desktop JDK 25 or later. */
internal object JvmBackend {
    val backend: EngineBackend? get() = null

    val default: KoblasEngine get() = BuiltinEngines.scalar

    @Suppress("UNUSED_PARAMETER", "FunctionOnlyReturningConstant") // Matches the desktop selection entry point.
    fun openBlas(only: Vendor?): Blas? = null
}
