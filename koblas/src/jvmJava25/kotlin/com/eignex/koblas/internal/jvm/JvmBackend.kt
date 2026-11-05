package com.eignex.koblas.internal.jvm

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.jvm25.host.*
import com.eignex.koblas.internal.jvm25.simd.*

/** Desktop implementation selected by the multi-release JAR on JDK 25 or later. */
internal object JvmBackend {
    val simd: BuiltinEngine? get() =
        if (simdAvailable) BuiltinEngine("jvm-simd", simdKernels) else null

    val default: KoblasEngine get() =
        standardEngine(EngineRuntime.Jvm, JvmBackend::class.java.module.isNativeAccessEnabled)

    fun open(vendor: Vendor): KoblasEngine? = JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)
}
