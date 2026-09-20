package com.eignex.koblas

import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.VendorVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter
import com.eignex.koblas.vendor.NativeVendorBlas

/** Built-in engines on Kotlin/Native. */
@KoblasEngineApi
public actual object BuiltinEngines {
    /** Pure Kotlin scalar at every level, without resolving a host library. */
    public actual val scalar: KoblasEngine by lazy {
        KoblasEngine(
            ScalarVectorKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
            null,
        )
    }

    /** The Vector API is a JVM module, so there is no distinct Native SIMD engine. */
    public actual val simd: KoblasEngine? = null
}

/**
 * The installed library where there is one, both for Level 1 above its width and for whole dense Level 2 and
 * 3 calls above theirs, and this library's portable Kotlin everywhere else.
 *
 * Portability does not depend on any of it. A host with no supported library gets [BuiltinEngines.scalar],
 * which computes every level in common Kotlin, and that is the floor rather than an alternative to it: the
 * composed engine below falls back to the same code for an operation the library does not export, a call too
 * small to pay for reaching it, a call whose no-read rules this library states and the standard leaves open,
 * and a product over operands packed for this library's own register tile, which no library has an argument
 * for. Which of the two a given call took is [KoblasEngine.routeOf]'s answer, not this function's.
 *
 * Why compose at all here: this target has no Vector API, LLVM will not reorder a floating-point reduction
 * on its own, and a bounds check and a safepoint poll are paid per element, so the portable arithmetic is
 * scalar and stays scalar. Against that, a tuned library is worth a foreign call and a pin per operand once
 * there is enough arithmetic to pay for them, which is the crossing
 * [com.eignex.koblas.dense.HostDensePolicy.MINIMUM_WORK] records.
 *
 * The JVM's default composes nothing, which is a policy rather than a measured comparison of the two at
 * Level 2 and 3. The matrix kernels this library owns there are its own Vector API ones, activated on their
 * own evidence, and a host binding stays explicitly callable beside them. No claim is made here about which
 * of those is faster at a given shape.
 *
 * The Level 1 kernels take the binding as its own type rather than as [com.eignex.koblas.vendor.Blas],
 * because they call the entry points that take a caller's array directly instead of a vector object. That is
 * the whole of the per-call cost at these widths. Opening a library on this runtime produces exactly that
 * type, so the check below never fails in practice; if some other implementation ever appears, Level 1 keeps
 * the portable kernels instead of paying for wrappers, and the dense composition still takes the binding.
 */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine {
    val vendor = selectedVendor ?: return BuiltinEngines.scalar
    val level1 = (vendor as? NativeVendorBlas)?.let { VendorVectorKernels(it) } ?: ScalarVectorKernels
    return KoblasEngine(
        level1,
        SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
        ScalarIndexedSparseKernels,
        vendor,
        hostDense = vendor,
    )
}
