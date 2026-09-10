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
    private val scalarEngine by lazy {
        KoblasContext(
            scalarDenseKernelFamilies,
            scalarSparseKernelFamilies,
        )
    }
    private val cEngine by lazy {
        if (CKernels.isAvailable) {
            KoblasContext(
                cDenseKernelFamilies,
                cSparseKernelFamilies,
            )
        } else {
            null
        }
    }
    private val simdEngine by lazy {
        if (SimdKernels.isAvailable) {
            KoblasContext(
                simdDenseKernelFamilies,
                simdSparseKernelFamilies,
            )
        } else {
            null
        }
    }

    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: KoblasContext
        get() = scalarEngine

    /** Bundled C kernels when the native library loaded successfully. */
    public actual val c: KoblasContext?
        get() = cEngine

    /** Vector API kernels when the incubator module resolved at startup. */
    public actual val simd: KoblasContext?
        get() = simdEngine
}
