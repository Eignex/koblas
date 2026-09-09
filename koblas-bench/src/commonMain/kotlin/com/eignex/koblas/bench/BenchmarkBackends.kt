package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinKernels
import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.engine
import com.eignex.koblas.koblas

internal const val BUILTIN_KERNELS = "built-in"
internal const val SCALAR_KERNELS = "scalar"
internal const val C_KERNELS = "c"
internal const val SIMD_KERNELS = "simd"

private fun reportResolution(arm: String, context: KoblasContext) {
    println()
    println(
        "resolved: arm=$arm vectorKernels=${context.vectorKernels.name} " +
            "sparseKernels=${context.sparseKernels.name} blas=${context.blas.name}",
    )
}

@OptIn(ExperimentalKoblasApi::class)
internal fun kernelEngine(arm: String): KoblasContext {
    val context = when (arm) {
        BUILTIN_KERNELS -> koblas
        else -> {
            val builtIn = when (arm) {
                SCALAR_KERNELS -> BuiltinKernels.scalar
                C_KERNELS -> BuiltinKernels.c
                SIMD_KERNELS -> BuiltinKernels.simd
                else -> error("unknown kernel arm: $arm")
            }
            checkNotNull(builtIn) { "the $arm kernel arm is unavailable on this platform" }.engine()
        }
    }
    if (arm != BUILTIN_KERNELS) {
        check(context.vectorKernels.name.startsWith(arm)) {
            "benchmark arm '$arm' resolved vector kernels to '${context.vectorKernels.name}'"
        }
    }
    reportResolution(arm, context)
    return context
}
