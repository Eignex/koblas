package com.eignex.koblas

import com.eignex.koblas.dense.CKernels
import com.eignex.koblas.dense.ScalarKernels
import com.eignex.koblas.dense.SimdKernels
import com.eignex.koblas.sparse.CSparseKernels
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.SimdSparseKernels

/** JVM built-in kernel providers. */
@ExperimentalKoblasApi
public actual object BuiltinKernels {
    private val scalarProvider by lazy {
        BuiltinKernelProvider(ScalarKernels, ReferenceSparseLinearAlgebra)
    }
    private val cProvider by lazy {
        if (CKernels.isAvailable) BuiltinKernelProvider(CKernels, CSparseKernels) else null
    }
    private val simdProvider by lazy {
        if (SimdKernels.isAvailable) BuiltinKernelProvider(SimdKernels, SimdSparseKernels) else null
    }

    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: BuiltinKernelProvider
        get() = scalarProvider

    /** Bundled C kernels when the native library loaded successfully. */
    public actual val c: BuiltinKernelProvider?
        get() = cProvider

    /** Vector API kernels when the incubator module resolved at startup. */
    public actual val simd: BuiltinKernelProvider?
        get() = simdProvider
}
