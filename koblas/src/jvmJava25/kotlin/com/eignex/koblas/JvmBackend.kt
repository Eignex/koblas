package com.eignex.koblas

import com.eignex.koblas.internal.jvm25.JvmVendorBlas
import com.eignex.koblas.internal.jvm25.JvmVendorLibrary
import com.eignex.koblas.internal.jvm25.SimdIndexedSparseKernels
import com.eignex.koblas.internal.jvm25.SimdPanelKernels
import com.eignex.koblas.internal.jvm25.SimdProductKernels
import com.eignex.koblas.internal.jvm25.SimdTriangularKernels
import com.eignex.koblas.internal.jvm25.SimdVectorKernels
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.hostPlatform

/** Desktop implementation selected by the multi-release JAR on JDK 25 or later. */
@OptIn(KoblasEngineApi::class)
internal object JvmBackend {
    // Species initialization must wait until the optional Vector API module has resolved.
    val simd: KoblasEngine? by lazy {
        if (SimdVectorKernels.isAvailable) {
            KoblasEngine(
                SimdVectorKernels,
                SimdIndexedSparseKernels,
                null,
                SimdPanelKernels,
                SimdProductKernels,
                SimdTriangularKernels,
                sparseKernelName = "simd-sparse",
            )
        } else {
            null
        }
    }

    fun openBlas(only: Vendor?): Blas? {
        val candidates = only?.let { listOf(it) } ?: Vendor.select(hostPlatform())
        return candidates.firstNotNullOfOrNull { vendor ->
            JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)
        }
    }
}
