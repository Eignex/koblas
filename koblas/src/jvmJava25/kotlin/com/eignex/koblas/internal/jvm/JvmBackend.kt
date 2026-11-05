package com.eignex.koblas.internal.jvm

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.engine.EngineRuntime
import com.eignex.koblas.engine.Vendor
import com.eignex.koblas.internal.engine.standardEngine
import com.eignex.koblas.internal.jvm25.host.JvmVendorBlas
import com.eignex.koblas.internal.jvm25.host.JvmVendorLibrary
import com.eignex.koblas.internal.jvm25.simd.simdAvailable
import com.eignex.koblas.internal.jvm25.simd.simdContributions
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.koblas

/** Desktop implementation selected by the multi-release JAR on JDK 25 or later. */
internal object JvmBackend {
    val contributions: List<EngineKernels>? get() =
        if (simdAvailable) simdContributions else null

    val default: KoblasEngine get() =
        standardEngine(EngineRuntime.Jvm, JvmBackend::class.java.module.isNativeAccessEnabled)

    fun open(vendor: Vendor): KoblasEngine? = JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)
}
