package com.eignex.koblas

import com.eignex.koblas.dense.F64ScalarKernels
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra

/** Built-in providers available while cross-compiling for a foreign Native host. */
@ExperimentalKoblasApi
public actual object F64BuiltinKernels {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: F64BuiltinKernelProvider =
        F64BuiltinKernelProvider(F64ScalarKernels, F64ReferenceSparseLinearAlgebra)

    /** C is unavailable without the target host's cinterop compilation. */
    public actual val c: F64BuiltinKernelProvider? = null

    /** SIMD is unavailable as a distinct Native provider. */
    public actual val simd: F64BuiltinKernelProvider? = null
}
