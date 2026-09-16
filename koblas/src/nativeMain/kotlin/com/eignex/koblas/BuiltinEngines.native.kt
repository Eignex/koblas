package com.eignex.koblas

import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.VendorVectorKernels
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

/**
 * The vendor Level 1 kernels where a library is installed, and the portable ones where none is.
 *
 * Level 2 and 3 already raise on a host with no library. Level 1 does not: it keeps working, which is what
 * makes the portable kernels the floor rather than an alternative.
 */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine = selectedVendor?.let { vendor ->
    KoblasEngine(
        VendorVectorKernels(vendor),
        SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
        ScalarIndexedSparseKernels,
        vendor,
    )
} ?: BuiltinEngines.scalar
