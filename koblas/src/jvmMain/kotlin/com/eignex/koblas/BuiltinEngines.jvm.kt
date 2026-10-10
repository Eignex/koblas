package com.eignex.koblas

import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.SimdPanelKernels
import com.eignex.koblas.dense.SimdProductKernels
import com.eignex.koblas.dense.SimdTriangularKernels
import com.eignex.koblas.dense.SimdVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SimdIndexedSparseKernels

/** JVM built-in engines. */
@KoblasEngineApi
public actual object BuiltinEngines {
    /** Pure Kotlin scalar at every level, without resolving a host library. */
    @get:JvmStatic
    public actual val scalar: KoblasEngine by lazy {
        KoblasEngine(
            ScalarVectorKernels,
            ScalarIndexedSparseKernels,
            null,
        )
    }

    /**
     * Owned Vector API kernels, also used by the JVM default when the module is available.
     * Matrix operations retain shared scheduling and report portable fallbacks through their routes.
     * Local calibration supports this selection on the measured host, not every architecture.
     *
     * Check availability before referencing [SimdPanelKernels]: its species initialization would
     * cause a linkage error on a runtime without the module.
     */
    @get:JvmStatic
    public actual val simd: KoblasEngine? by lazy {
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
}

/** The Vector API composition where the module resolved, and the portable engine where it did not. */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine = BuiltinEngines.simd ?: BuiltinEngines.scalar
