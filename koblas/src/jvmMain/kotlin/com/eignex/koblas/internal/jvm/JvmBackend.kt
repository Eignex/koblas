package com.eignex.koblas.internal.jvm

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.*

/** Portable entry points replaced by the multi-release JAR on desktop JDK 25 or later. */
internal object JvmBackend {
    val simd: BuiltinEngine? get() = null

    val default: KoblasEngine get() = KoblasEngine.scalar

    @Suppress("UNUSED_PARAMETER", "FunctionOnlyReturningConstant") // Matches the desktop selection entry point.
    fun open(vendor: Vendor): KoblasEngine? = null
}
