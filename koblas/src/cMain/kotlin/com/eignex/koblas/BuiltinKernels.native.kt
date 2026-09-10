package com.eignex.koblas

import com.eignex.koblas.dense.platformDenseKernelFamilies
import com.eignex.koblas.dense.scalarDenseKernelFamilies
import com.eignex.koblas.sparse.nativeCSparseKernelFamilies
import com.eignex.koblas.sparse.scalarSparseKernelFamilies

/** Kotlin/Native built-in kernel providers. */
public actual object BuiltinKernels {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: KoblasContext =
        KoblasContext(
            scalarDenseKernelFamilies,
            scalarSparseKernelFamilies,
        )

    /** C kernels compiled into this Native artifact. */
    public actual val c: KoblasContext? =
        KoblasContext(
            platformDenseKernelFamilies,
            nativeCSparseKernelFamilies,
        )

    /** SIMD is unavailable as a distinct Native provider. */
    public actual val simd: KoblasContext? = null
}
