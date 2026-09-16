package com.eignex.koblas

import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.SimdVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SimdIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter

/** JVM built-in engines. */
@KoblasEngineApi
public actual object BuiltinEngines {
    /** Pure Kotlin scalar Level 1 beside the selected vendor. */
    @get:JvmStatic
    public actual val scalar: KoblasEngine by lazy {
        KoblasEngine(
            ScalarVectorKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
            selectedVendor,
        )
    }

    /** Vector API Level 1 when the incubator module resolved at startup, beside the selected vendor. */
    @get:JvmStatic
    public actual val simd: KoblasEngine? by lazy {
        if (SimdVectorKernels.isAvailable) {
            KoblasEngine(
                SimdVectorKernels,
                SparseKernelAdapter("simd-sparse", SimdVectorKernels, SimdIndexedSparseKernels),
                SimdIndexedSparseKernels,
                selectedVendor,
            )
        } else {
            null
        }
    }
}
