package com.eignex.koblas

import com.eignex.koblas.dense.scalarDenseKernelFamilies
import com.eignex.koblas.sparse.scalarSparseKernelFamilies

/** Built-in providers available while cross-compiling for a foreign Native host. */
public actual object BuiltinKernels {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: BuiltinKernelProvider =
        BuiltinKernelProvider(
            scalarDenseKernelFamilies,
            scalarSparseKernelFamilies,
        )

    /** C is unavailable without the target host's cinterop compilation. */
    public actual val c: BuiltinKernelProvider? = null

    /** SIMD is unavailable as a distinct Native provider. */
    public actual val simd: BuiltinKernelProvider? = null
}
