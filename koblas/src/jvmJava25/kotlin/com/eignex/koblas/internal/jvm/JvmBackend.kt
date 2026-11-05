package com.eignex.koblas.internal.jvm

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.engine.standardEngine
import com.eignex.koblas.internal.jvm25.host.*
import com.eignex.koblas.internal.jvm25.simd.*
import com.eignex.koblas.internal.kernels.EngineKernels

/** Desktop implementation selected by the multi-release JAR on JDK 25 or later. */
internal object JvmBackend {
    val contributions: List<EngineKernels>? get() =
        if (simdAvailable) simdContributions else null

    val default: KoblasEngine get() =
        standardEngine(EngineRuntime.Jvm, JvmBackend::class.java.module.isNativeAccessEnabled)

    fun open(vendor: Vendor): KoblasEngine? = JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)
}
