package com.eignex.koblas

import com.eignex.koblas.dense.PortablePackedKernels
import com.eignex.koblas.dense.ScalarPanelKernels
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter

/** Built-in engines available while cross-compiling for a foreign Native host. */
public actual object BuiltinEngines {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: KoblasEngine =
        KoblasEngine(
            ScalarVectorKernels,
            ScalarPanelKernels,
            PortablePackedKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
        )

    /** C is unavailable without the target host's cinterop compilation. */
    public actual val c: KoblasEngine? = null

    /** SIMD is unavailable as a distinct Native engine. */
    public actual val simd: KoblasEngine? = null

    /** No native archive is available in this scalar cross-target artifact. */
    public actual val nativeVariants: List<NativeVariant> = emptyList()

    /** Exact native execution requires a target archive. */
    public actual fun exactC(variant: NativeVariant): KoblasEngine =
        throw IllegalArgumentException("native ${variant.name} is not built for this target")
}
