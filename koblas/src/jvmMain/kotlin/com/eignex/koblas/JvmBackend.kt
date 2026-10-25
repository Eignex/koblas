package com.eignex.koblas

import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.Vendor

/** Portable entry points replaced by the multi-release JAR on desktop JDK 25 or later. */
@OptIn(KoblasEngineApi::class)
internal object JvmBackend {
    val simd: KoblasEngine? get() = null

    @Suppress("UNUSED_PARAMETER", "FunctionOnlyReturningConstant") // Matches the desktop selection entry point.
    fun openBlas(only: Vendor?): Blas? = null
}
