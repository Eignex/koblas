package com.eignex.koblas

import com.eignex.koblas.internal.jvm25.host.JvmVendorBlas
import com.eignex.koblas.internal.jvm25.host.JvmVendorLibrary
import com.eignex.koblas.internal.jvm25.simd.simdAvailable
import com.eignex.koblas.internal.jvm25.simd.simdContributions
import com.eignex.koblas.vendor.Vendor

/** Desktop implementation selected by the multi-release JAR on JDK 25 or later. */
internal object JvmBackend {
    val contributions: List<KoblasEngine>? get() =
        if (simdAvailable) simdContributions else null

    val default: KoblasEngine by lazy {
        standardEngine("jvm", JvmBackend::class.java.module.isNativeAccessEnabled)
    }

    fun open(vendor: Vendor): KoblasEngine? = JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)
}
