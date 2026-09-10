package com.eignex.koblas

import com.eignex.koblas.dense.CKernels
import com.eignex.koblas.dense.SimdKernels
import com.eignex.koblas.dense.cDenseKernelFamilies
import com.eignex.koblas.dense.scalarDenseKernelFamilies
import com.eignex.koblas.dense.simdDenseKernelFamilies
import com.eignex.koblas.sparse.cSparseKernelFamilies
import com.eignex.koblas.sparse.scalarSparseKernelFamilies
import com.eignex.koblas.sparse.simdSparseKernelFamilies

/** JVM built-in kernel providers. */
public actual object BuiltinKernels {
    private val scalarProvider by lazy {
        BuiltinKernelProvider(
            scalarDenseKernelFamilies,
            scalarSparseKernelFamilies,
        )
    }
    private val cProvider by lazy {
        if (CKernels.isAvailable) {
            BuiltinKernelProvider(
                cDenseKernelFamilies,
                cSparseKernelFamilies,
            )
        } else {
            null
        }
    }
    private val simdProvider by lazy {
        if (SimdKernels.isAvailable) {
            BuiltinKernelProvider(
                simdDenseKernelFamilies,
                simdSparseKernelFamilies,
            )
        } else {
            null
        }
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
