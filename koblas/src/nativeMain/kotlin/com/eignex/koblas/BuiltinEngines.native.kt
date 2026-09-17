package com.eignex.koblas

import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.VendorVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter
import com.eignex.koblas.vendor.NativeVendorBlas

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
 *
 * The Level 1 kernels take the binding as its own type rather than as [com.eignex.koblas.vendor.Blas],
 * because they call the entry points that take a caller's array directly instead of a vector object. That is
 * the whole of the per-call cost at these widths. Opening a library on this runtime produces exactly that
 * type, so the check below never fails in practice; if some other implementation ever appears, Level 1 keeps
 * the portable kernels instead of paying for wrappers.
 */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine {
    val vendor = selectedVendor as? NativeVendorBlas ?: return BuiltinEngines.scalar
    return KoblasEngine(
        VendorVectorKernels(vendor),
        SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
        ScalarIndexedSparseKernels,
        vendor,
    )
}
