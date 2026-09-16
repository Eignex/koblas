package com.eignex.koblas

import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter

/** Built-in engines on Kotlin/Native. */
@KoblasEngineApi
public actual object BuiltinEngines {
    /** Pure Kotlin scalar Level 1 beside the selected vendor. */
    public actual val scalar: KoblasEngine by lazy {
        KoblasEngine(
            ScalarVectorKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
            selectedVendor,
        )
    }

    /** The Vector API is a JVM module, so there is no distinct Native SIMD engine. */
    public actual val simd: KoblasEngine? = null
}
