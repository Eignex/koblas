package com.eignex.koblas

import com.eignex.koblas.internal.jvm25.host.JvmVendorBlas
import com.eignex.koblas.internal.jvm25.host.JvmVendorLibrary
import com.eignex.koblas.internal.jvm25.simd.simdAvailable
import com.eignex.koblas.internal.jvm25.simd.simdContributions
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.hostPlatform

/** Desktop implementation selected by the multi-release JAR on JDK 25 or later. */
internal object JvmBackend {
    val backend: BackendContributions? get() =
        if (simdAvailable) simdContributions else null

    val default: KoblasEngine by lazy {
        standardEngine("jvm", JvmBackend::class.java.module.isNativeAccessEnabled)
    }

    fun openBlas(only: Vendor?): Blas? {
        val candidates = only?.let { listOf(it) } ?: Vendor.select(hostPlatform())
        return candidates.firstNotNullOfOrNull { vendor ->
            JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)
        }
    }
}
