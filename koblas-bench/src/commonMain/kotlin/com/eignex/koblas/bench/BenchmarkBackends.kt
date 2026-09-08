package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra

internal const val REFERENCE_BACKEND = "reference"
internal const val HOST_BACKEND = "host"
internal const val AUTOMATIC_BACKEND = "automatic"
internal const val AUTOMATIC_KERNELS = "automatic"
internal const val SCALAR_KERNELS = "scalar"
internal const val C_KERNELS = "c"

/**
 * The Vector API kernels. Absent from every `@Param` list because Kotlin/Native has no such provider and a
 * benchmark configuration covers all targets, so a full native run would ask for a provider that cannot
 * exist. Pass it explicitly on the JVM with `-Pbench.param.kernels=simd`.
 */
internal const val SIMD_KERNELS = "simd"

/**
 * Fails when an arm installed something other than what its name promises.
 *
 * An arm that quietly resolves elsewhere is worse than one that cannot run: the benchmark still produces a
 * plausible table, and every number in it is attributed to an implementation that never executed. The
 * comparison that motivated this check asked for the portable kernels against a host and received the host
 * on both sides, because discovery outranks the built-in providers wherever a host library is installed.
 */
private fun requireResolved(arm: String, half: String, resolved: String, expected: String) {
    check(resolved.matchesExpectation(expected)) {
        "benchmark arm '$arm' resolved $half to '$resolved', but that arm names $expected. " +
            "Results from this run would credit an implementation that did not execute."
    }
}

/**
 * True when a resolved name answers to [expected]. The SIMD kernels append a lane count to their name and
 * the routed kernels join a compiled-in half to a host one with `+`, so neither compares by equality.
 */
private fun String.matchesExpectation(expected: String): Boolean = when (expected) {
    SIMD_KERNELS -> startsWith(SIMD_KERNELS) && !contains('+')
    else -> this == expected
}

/**
 * Prints what an arm resolved to, one line per installed arm.
 *
 * The `resolved:` prefix is what `report.sh` collects into a report's metadata, so it stays at the front of
 * the line. The arm is named alongside the halves because a bare list of backends does not say which
 * request produced it, which is exactly the confusion this whole check exists to prevent.
 */
private fun reportResolution(arm: String, vararg halves: Pair<String, String>) {
    // The leading newline is load-bearing. A forked JMH process renders its first burst of output as
    // decimal character codes until a line break syncs it, and a resolution line caught in that burst is
    // unreadable and invisible to report.sh, which collects these lines by their prefix.
    println()
    println("resolved: arm=$arm " + halves.joinToString(" ") { "${it.first}=${it.second}" })
}

/**
 * The sparse factorization half has no host provider in koblas-bench, so `automatic` and `reference` are the
 * two arms and a request for `host` fails rather than silently measuring the portable code.
 */
internal fun installSparseDecompositionBackend(backend: String) {
    installBackends(null)
    when (backend) {
        AUTOMATIC_BACKEND -> discoverBackends()
        REFERENCE_BACKEND -> installBackends(koblas.with(sparseDecompositions = F64ReferenceSparseLinearAlgebra))
        HOST_BACKEND -> error("the host sparse decomposition backend is unavailable")
        else -> error("unknown backend: $backend")
    }
    val resolved = koblas.sparseDecompositions.name
    if (backend == REFERENCE_BACKEND) requireResolved(backend, "sparseDecompositions", resolved, REFERENCE_BACKEND)
    reportResolution(backend, "sparseDecompositions" to resolved)
}

/** The sparse BLAS half has no host provider either, so this mirrors [installSparseDecompositionBackend]. */
internal fun installSparseBlasBackend(backend: String) {
    installBackends(null)
    when (backend) {
        AUTOMATIC_BACKEND -> discoverBackends()
        REFERENCE_BACKEND -> installBackends(ContextBuilder().resolve())
        HOST_BACKEND -> error("the host sparse BLAS backend is unavailable")
        else -> error("unknown backend: $backend")
    }
    val resolved = koblas.sparseBlas.name
    if (backend == REFERENCE_BACKEND) requireResolved(backend, "sparseBlas", resolved, REFERENCE_BACKEND)
    reportResolution(backend, "sparseBlas" to resolved, "sparseKernels" to koblas.sparseKernels.name)
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
        REFERENCE_BACKEND -> installBackends(ContextBuilder().resolve())
        HOST_BACKEND -> {
            discoverBackends()
            check(koblas.basisSolvers.name != REFERENCE_BACKEND) { "the host basis solver backend is unavailable" }
        }
        else -> error("unknown backend: $backend")
    }
    val resolved = koblas.basisSolvers.name
    if (backend == REFERENCE_BACKEND) requireResolved(backend, "basisSolvers", resolved, REFERENCE_BACKEND)
    reportResolution(backend, "basisSolvers" to resolved)
}

/**
 * Installs the kernel arm named by [provider].
 *
 * A pinned provider is resolved from a portable seed rather than from the installed context. Seeded from
 * the installed one it inherited whatever discovery had chosen for the matrix halves, so a level-2 routine
 * under a `scalar` arm still ran on a host library and the kernel arms measured the same code.
 *
 * There is no host arm here. koblas no longer carries a level-1 host binding, so the only kernels a run can
 * name are its own, and a level-1 comparison against a host library is not something this module can make.
 */
@OptIn(ExperimentalKoblasApi::class)
internal fun installKernelProvider(provider: String) {
    installBackends(null)
    when (provider) {
        AUTOMATIC_KERNELS -> discoverBackends()
        else -> {
            val builtIn = when (provider) {
                SCALAR_KERNELS -> F64BuiltinKernels.scalar
                C_KERNELS -> F64BuiltinKernels.c
                SIMD_KERNELS -> F64BuiltinKernels.simd
                else -> error("unknown kernel provider: $provider")
            }
            checkNotNull(builtIn) { "the $provider kernel provider is unavailable on this platform" }
            installBackends(ContextBuilder().withBuiltinKernels(builtIn).resolve())
        }
    }
    val kernels = koblas.kernels.name
    when (provider) {
        SCALAR_KERNELS, C_KERNELS, SIMD_KERNELS -> {
            requireResolved(provider, "kernels", kernels, provider)
            // The matrix halves matter to a kernel arm because some suites under it measure a level-2
            // routine, which reaches the kernels only through the BLAS half above them.
            requireResolved(provider, "blas", koblas.blas.name, REFERENCE_BACKEND)
        }
    }
    reportResolution(
        provider,
        "kernels" to kernels,
        "sparseKernels" to koblas.sparseKernels.name,
        "blas" to koblas.blas.name,
    )
}
