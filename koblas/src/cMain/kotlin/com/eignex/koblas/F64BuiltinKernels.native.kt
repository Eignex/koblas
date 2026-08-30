package com.eignex.koblas

import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.dense.F64ScalarKernels
import com.eignex.koblas.sparse.F64PlatformSparseKernels
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra

/** Kotlin/Native built-in kernel providers. */
@ExperimentalKoblasApi
public actual object F64BuiltinKernels {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: F64BuiltinKernelProvider =
        F64BuiltinKernelProvider(F64ScalarKernels, F64ReferenceSparseLinearAlgebra)

    /** C kernels compiled into this Native artifact. */
    public actual val c: F64BuiltinKernelProvider? =
        F64BuiltinKernelProvider(F64PlatformKernels, F64PlatformSparseKernels)

    /** SIMD is unavailable as a distinct Native provider. */
    public actual val simd: F64BuiltinKernelProvider? = null
}
