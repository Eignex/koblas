package com.eignex.koblas

import com.eignex.koblas.dense.NativeCPackedKernels
import com.eignex.koblas.dense.NativeCPanelKernels
import com.eignex.koblas.dense.NativeCVectorKernels
import com.eignex.koblas.dense.PortablePackedKernels
import com.eignex.koblas.dense.ScalarPanelKernels
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.NativeCIndexedSparseKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter

/** Kotlin/Native built-in engines. */
public actual object BuiltinEngines {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: KoblasContext =
        KoblasContext(
            ScalarVectorKernels,
            ScalarPanelKernels,
            PortablePackedKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
        )

    /** C kernels compiled into this Native artifact. */
    public actual val c: KoblasContext? =
        KoblasContext(
            NativeCVectorKernels,
            NativeCPanelKernels,
            NativeCPackedKernels,
            SparseKernelAdapter("c-sparse", NativeCVectorKernels, NativeCIndexedSparseKernels),
            NativeCIndexedSparseKernels,
        )

    /** SIMD is unavailable as a distinct Native engine. */
    public actual val simd: KoblasContext? = null
}
