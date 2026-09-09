package com.eignex.koblas

import com.eignex.koblas.dense.platformDenseKernelFamilies
import com.eignex.koblas.dense.scalarDenseKernelFamilies
import com.eignex.koblas.sparse.PlatformSparseKernels
import com.eignex.koblas.sparse.ScalarSparseKernels

/** Kotlin/Native built-in kernel providers. */
@ExperimentalKoblasApi
public actual object BuiltinKernels {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: BuiltinKernelProvider =
        BuiltinKernelProvider(
            scalarDenseKernelFamilies,
            ScalarSparseKernels,
        )

    /** C kernels compiled into this Native artifact. */
    public actual val c: BuiltinKernelProvider? =
        BuiltinKernelProvider(
            platformDenseKernelFamilies,
            PlatformSparseKernels,
        )

    /** SIMD is unavailable as a distinct Native provider. */
    public actual val simd: BuiltinKernelProvider? = null
}
