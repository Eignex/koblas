package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra

internal const val REFERENCE_BACKEND = "reference"
internal const val HOST_BACKEND = "host"
internal const val AUTOMATIC_KERNELS = "automatic"
internal const val SCALAR_KERNELS = "scalar"
internal const val C_KERNELS = "c"
internal const val SIMD_KERNELS = "simd"

internal expect fun useHost(): Boolean

internal expect fun useSparseLu(): Boolean

internal expect fun useSparseProduct(): Boolean

internal fun installBackend(backend: String) {
    // Cleared first, so an arm starts from the portable halves rather than whatever the previous one left.
    installBackends(null)
    // Each branch installs before reporting: an interpolation reads koblasInfo before the call beside it
    // runs, which would describe the state the arm was replacing.
    val installed = when (backend) {
        HOST_BACKEND -> useHost()
        else -> {
            installBackends(
                koblas.with(blas = F64ReferenceLinearAlgebra, decompositions = F64ReferenceLinearAlgebra),
            )
            true
        }
    }
    println("resolved: $koblasInfo (installed=$installed)")
}

@OptIn(ExperimentalKoblasApi::class)
internal fun installKernelProvider(provider: String) {
    installBackends(null)
    if (provider == AUTOMATIC_KERNELS) return
    if (provider == HOST_BACKEND) {
        check(useHost()) { "the host kernel provider is unavailable" }
        return
    }
    val builtIn = when (provider) {
        SCALAR_KERNELS -> F64BuiltinKernels.scalar
        C_KERNELS -> F64BuiltinKernels.c
        SIMD_KERNELS -> F64BuiltinKernels.simd
        else -> error("unknown kernel provider: $provider")
    }
    checkNotNull(builtIn) { "$provider kernel provider is unavailable" }
    installBackends(F64ContextBuilder(koblas).withBuiltinKernels(builtIn).resolve())
}
