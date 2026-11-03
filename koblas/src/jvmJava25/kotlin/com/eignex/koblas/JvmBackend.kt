package com.eignex.koblas

import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.Vendor
import com.eignex.koblas.internal.jvm25.host.JvmVendorBlas
import com.eignex.koblas.internal.jvm25.host.JvmVendorLibrary
import com.eignex.koblas.internal.jvm25.simd.SimdPanelKernels
import com.eignex.koblas.internal.jvm25.simd.SimdProductKernels
import com.eignex.koblas.internal.jvm25.simd.SimdTriangularKernels
import com.eignex.koblas.internal.jvm25.simd.simdAvailable
import com.eignex.koblas.internal.jvm25.simd.simdContributions

/** Desktop implementation selected by the multi-release JAR on JDK 25 or later. */
internal object JvmBackend {
    val contributions: List<EngineKernels>? get() =
        if (simdAvailable) simdContributions else null

    val default: KoblasEngine by lazy {
        standardEngine(EngineRuntime.Jvm, JvmBackend::class.java.module.isNativeAccessEnabled)
    }

    fun open(vendor: Vendor): KoblasEngine? = JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)

    fun panelImplementation(
        engine: KoblasEngine,
        work: PanelWork,
        rows: Int,
        columns: Int,
        contiguous: Boolean,
    ): String? = if (simdAvailable && engine === SimdPanelKernels) {
        SimdPanelKernels.implementationFor(
            work,
            rows,
            columns,
            contiguous,
        )
    } else {
        null
    }
    fun productImplementations(engine: KoblasEngine, rows: Int, columns: Int, depth: Int): List<String>? =
        if (simdAvailable && engine === SimdProductKernels) {
            SimdProductKernels.implementationsFor(
                rows,
                columns,
                depth,
            )
        } else {
            null
        }
    fun triangularImplementations(engine: KoblasEngine, order: Int, sides: Int, contiguous: Boolean): List<String>? =
        if (simdAvailable && engine === SimdTriangularKernels) {
            SimdTriangularKernels.implementationsFor(
                order,
                sides,
                contiguous,
            )
        } else {
            null
        }
}
