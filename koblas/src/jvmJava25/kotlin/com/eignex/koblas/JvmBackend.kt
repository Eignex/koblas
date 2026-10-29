package com.eignex.koblas

import com.eignex.koblas.internal.jvm25.host.JvmHostDensePolicy
import com.eignex.koblas.internal.jvm25.host.JvmVendorBlas
import com.eignex.koblas.internal.jvm25.host.JvmVendorLibrary
import com.eignex.koblas.internal.jvm25.simd.SimdIndexedSparseKernels
import com.eignex.koblas.internal.jvm25.simd.SimdPanelKernels
import com.eignex.koblas.internal.jvm25.simd.SimdProductKernels
import com.eignex.koblas.internal.jvm25.simd.SimdTriangularKernels
import com.eignex.koblas.internal.jvm25.simd.SimdVectorKernels
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.hostPlatform

/** Desktop implementation selected by the multi-release JAR on JDK 25 or later. */
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

    val default: KoblasEngine by lazy {
        val exact = simd ?: BuiltinEngines.scalar
        // Resolving a library is opt-in native access; pure JVM callers keep the portable composition.
        if (!JvmBackend::class.java.module.isNativeAccessEnabled ||
            Vendor.select(hostPlatform()).firstOrNull() != Vendor.OneMkl
        ) {
            exact
        } else {
            val host = openBlas(Vendor.OneMkl)
            if (host == null) {
                exact
            } else {
                KoblasEngine(
                    exact.vectorKernels,
                    exact.indexedSparseKernels,
                    host,
                    exact.panelKernels,
                    exact.productKernels,
                    exact.triangularKernels,
                    sparseKernelName = exact.sparseKernels.name,
                    nativeEligible = JvmHostDensePolicy,
                )
            }
        }
    }

    fun openBlas(only: Vendor?): Blas? {
        val candidates = only?.let { listOf(it) } ?: Vendor.select(hostPlatform())
        return candidates.firstNotNullOfOrNull { vendor ->
            JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)
        }
    }
}
