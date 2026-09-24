package com.eignex.koblas

import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter

/** Built-in engines on Android. */
@KoblasEngineApi
public actual object BuiltinEngines {
    /** Pure Kotlin scalar at every level, without resolving a host library. */
    @get:JvmStatic
    public actual val scalar: KoblasEngine by lazy {
        KoblasEngine(
            ScalarVectorKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
            null,
        )
    }

    /** ART does not ship the Vector API, so there is no distinct Android SIMD engine. */
    @get:JvmStatic
    public actual val simd: KoblasEngine? = null
}

/**
 * The portable engine. The bundled OpenBLAS is reachable through [com.eignex.koblas.vendor.openBlas], and the
 * default composes nothing until its crossover on a device is established.
 */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine = BuiltinEngines.scalar
