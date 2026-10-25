package com.eignex.koblas

import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels

/** JVM built-in engines. */
@KoblasEngineApi
public actual object BuiltinEngines {
    /** Pure Kotlin scalar at every level, without resolving a host library. */
    @get:JvmStatic
    public actual val scalar: KoblasEngine by lazy {
        KoblasEngine(
            ScalarVectorKernels,
            ScalarIndexedSparseKernels,
            null,
        )
    }

    /** Vector API kernels on desktop JDK 25 or later, or null on portable runtimes. */
    @get:JvmStatic
    public actual val simd: KoblasEngine? get() = JvmBackend.simd
}

/** The Vector API composition where the module resolved, and the portable engine where it did not. */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine = BuiltinEngines.simd ?: BuiltinEngines.scalar
