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
 * The bundled OpenBLAS for whole dense Level 2 and 3 calls above [com.eignex.koblas.dense.HostDensePolicy]'s
 * size, and this library's portable Kotlin for everything else, or [BuiltinEngines.scalar] where the bundled
 * library did not load.
 *
 * The same composition as Kotlin/Native. Reaching the library through JNI has a fixed cost per call that the
 * size policy is there to pay for, and Level 1 stays portable because a short vector call finishes in less time
 * than that cost alone. Which of the two a given call took is [KoblasEngine.routeOf]'s answer, not this one's.
 */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine {
    val vendor = selectedVendor ?: return BuiltinEngines.scalar
    return KoblasEngine(
        ScalarVectorKernels,
        SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
        ScalarIndexedSparseKernels,
        vendor,
        hostDense = vendor,
    )
}
