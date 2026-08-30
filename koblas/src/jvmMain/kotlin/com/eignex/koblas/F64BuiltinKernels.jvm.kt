package com.eignex.koblas

import com.eignex.koblas.dense.F64CKernels
import com.eignex.koblas.dense.F64ScalarKernels
import com.eignex.koblas.dense.F64SimdKernels
import com.eignex.koblas.sparse.F64CSparseKernels
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SimdSparseKernels

/** JVM built-in kernel providers. */
@ExperimentalKoblasApi
public actual object F64BuiltinKernels {
    private val scalarProvider by lazy {
        F64BuiltinKernelProvider(F64ScalarKernels, F64ReferenceSparseLinearAlgebra)
    }
    private val cProvider by lazy {
        if (F64CKernels.isAvailable) F64BuiltinKernelProvider(F64CKernels, F64CSparseKernels) else null
    }
    private val simdProvider by lazy {
        if (F64SimdKernels.isAvailable) F64BuiltinKernelProvider(F64SimdKernels, F64SimdSparseKernels) else null
    }

    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: F64BuiltinKernelProvider
        get() = scalarProvider

    /** Bundled C kernels when the native library loaded successfully. */
    public actual val c: F64BuiltinKernelProvider?
        get() = cProvider

    /** Vector API kernels when the incubator module resolved at startup. */
    public actual val simd: F64BuiltinKernelProvider?
        get() = simdProvider
}
