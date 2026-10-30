package com.eignex.koblas

import com.eignex.koblas.internal.dense.ScalarVectorKernels
import com.eignex.koblas.internal.host.NativeBlasPolicy
import com.eignex.koblas.internal.host.NativeVendorBlas
import com.eignex.koblas.internal.host.VendorVectorKernels
import com.eignex.koblas.internal.sparse.ScalarIndexedSparseKernels

/** Built-in engines on Kotlin/Native. */
public actual object BuiltinEngines {
    /** Pure Kotlin scalar at every level, without resolving a host library. */
    public actual val scalar: KoblasEngine get() = scalarEngine

    /** The Vector API is a JVM module, so there is no distinct Native SIMD engine. */
    public actual val simd: KoblasEngine? = null
}

/**
 * Compose installed vendor kernels with portable Kotlin fallbacks. Level 1 and whole dense
 * Level 2 and 3 calls use the vendor above their thresholds. Without a supported library,
 * [BuiltinEngines.scalar] provides every level.
 *
 * Small calls, missing exports, no-read contracts and locally packed product tiles use portable
 * kernels. [KoblasEngine.routeOf] reports the route each call actually takes.
 *
 * Kotlin/Native reductions remain scalar, so sufficiently large vendor calls amortize pinning
 * and foreign-call overhead. The JVM uses its owned Vector API kernels under a separate default
 * policy; this does not compare their performance against vendor matrix calls.
 *
 * Level 1 needs [com.eignex.koblas.internal.host.NativeVendorBlas]'s raw-array entry points to avoid
 * operand wrappers on short calls. A different binding keeps portable Level 1 kernels while
 * remaining available to the dense composition.
 */
internal actual fun platformEngine(): KoblasEngine {
    val vendor = selectedVendor ?: return BuiltinEngines.scalar
    val level1 = (vendor as? NativeVendorBlas)?.let { VendorVectorKernels(it) } ?: ScalarVectorKernels
    return KoblasEngine(
        level1,
        ScalarIndexedSparseKernels,
        vendor,
        nativeEligible = NativeBlasPolicy,
    )
}
