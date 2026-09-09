package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinKernels
import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.engine
import com.eignex.koblas.koblas

internal const val AUTOMATIC_KERNELS = "automatic"
internal const val SCALAR_KERNELS = "scalar"
internal const val C_KERNELS = "c"
internal const val SIMD_KERNELS = "simd"

private fun reportResolution(arm: String, context: KoblasContext) {
    println()
    println(
        "resolved: arm=$arm kernels=${context.kernels.name} " +
            "sparseKernels=${context.sparseKernels.name} blas=${context.blas.name}",
    )
}

@OptIn(ExperimentalKoblasApi::class)
internal fun kernelEngine(provider: String): KoblasContext {
    val context = when (provider) {
        AUTOMATIC_KERNELS -> koblas
        else -> {
            val builtIn = when (provider) {
                SCALAR_KERNELS -> BuiltinKernels.scalar
                C_KERNELS -> BuiltinKernels.c
                SIMD_KERNELS -> BuiltinKernels.simd
                else -> error("unknown kernel provider: $provider")
            }
            checkNotNull(builtIn) { "the $provider kernel provider is unavailable on this platform" }.engine()
        }
    }
    if (provider != AUTOMATIC_KERNELS) {
        check(context.kernels.name.startsWith(provider)) {
            "benchmark arm '$provider' resolved kernels to '${context.kernels.name}'"
        }
    }
    reportResolution(provider, context)
    return context
}
