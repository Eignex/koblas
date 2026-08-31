package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra

internal const val REFERENCE_BACKEND = "reference"
internal const val HOST_BACKEND = "host"
internal const val AUTOMATIC_BACKEND = "automatic"
internal const val AUTOMATIC_KERNELS = "automatic"
internal const val SCALAR_KERNELS = "scalar"
internal const val C_KERNELS = "c"
internal const val SIMD_KERNELS = "simd"

internal expect fun useHost(): Boolean

internal expect fun useSparseLu(): Boolean

internal expect fun useSparseProduct(): Boolean

internal fun installDenseBackend(backend: String) {
    installBackends(null)
    when (backend) {
        AUTOMATIC_BACKEND -> discoverBackends()
        REFERENCE_BACKEND ->
            installBackends(
                koblas.with(blas = F64ReferenceLinearAlgebra, decompositions = F64ReferenceLinearAlgebra),
            )
        HOST_BACKEND -> check(useHost()) { "the host dense backend is unavailable" }
        else -> error("unknown backend: $backend")
    }
    println("resolved: $koblasInfo")
}

internal fun installSparseDecompositionBackend(backend: String) {
    installBackends(null)
    when (backend) {
        AUTOMATIC_BACKEND -> discoverBackends()
        REFERENCE_BACKEND -> installBackends(koblas.with(sparseDecompositions = F64ReferenceSparseLinearAlgebra))
        HOST_BACKEND -> check(useSparseLu()) { "the host sparse decomposition backend is unavailable" }
        else -> error("unknown backend: $backend")
    }
    println("resolved: sparseDecompositions=${koblas.sparseDecompositions.name}")
}

internal fun installSparseBlasBackend(backend: String) {
    installBackends(null)
    when (backend) {
        AUTOMATIC_BACKEND -> discoverBackends()
        REFERENCE_BACKEND -> Unit
        HOST_BACKEND -> check(useSparseProduct()) { "the host sparse BLAS backend is unavailable" }
        else -> error("unknown backend: $backend")
    }
    println("resolved: sparseBlas=${koblas.sparseBlas.name}")
}

/**
 * Basis solvers are their own half, distinct from [installSparseDecompositionBackend]'s
 * [koblas.sparseDecompositions]: [koblas-bench] carries no `koblas-hfactor` dependency, so a host
 * implementation surfaces only if automatic discovery finds one already on the classpath.
 */
internal fun installBasisSolverBackend(backend: String) {
    installBackends(null)
    when (backend) {
        AUTOMATIC_BACKEND -> discoverBackends()
        REFERENCE_BACKEND -> Unit
        HOST_BACKEND -> {
            discoverBackends()
            check(koblas.basisSolvers.name != REFERENCE_BACKEND) { "the host basis solver backend is unavailable" }
        }
        else -> error("unknown backend: $backend")
    }
    println("resolved: basisSolvers=${koblas.basisSolvers.name}")
}

@OptIn(ExperimentalKoblasApi::class)
internal fun installKernelProvider(provider: String) {
    installBackends(null)
    if (provider == AUTOMATIC_KERNELS) {
        discoverBackends()
        return
    }
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
