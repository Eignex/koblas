package com.eignex.koblas

import com.eignex.koblas.hostblas.HostBlas
import com.eignex.koblas.hostblas.HostBlasCalls
import com.eignex.koblas.hostblas.HostLapack
import java.util.ServiceLoader

/**
 * JVM backend seam. koblas's own host-OpenBLAS backend comes first when the machine has the library
 * installed; after that, [LinearAlgebra] providers discovered via [ServiceLoader], so a third-party
 * backend artifact on the classpath activates with no code changes. Each candidate is probed with a tiny
 * computation before being accepted, so one whose native library fails to load on the current platform is
 * skipped rather than crashing startup.
 *
 * When several providers are on the classpath the highest [LinearAlgebra.priority] that passes the
 * probe wins.
 *
 * The `koblas.backend` system property overrides discovery: `reference` forces the portable
 * [ReferenceLinearAlgebra]; any other value selects the provider whose [LinearAlgebra.name] matches,
 * falling back to the reference when none does. The property is read once, when [koblas] initializes.
 */
actual fun platformLinearAlgebra(): LinearAlgebra? {
    val requested = System.getProperty("koblas.backend")
    if (requested == "reference") return null
    for (provider in loadProviders().sortedByDescending { it.priority }) {
        if (requested != null && provider.name != requested) continue
        if (probe(provider)) return provider
    }
    return null
}

/**
 * Registers koblas's built-in host-BLAS halves when the machine provides them.
 *
 * Called from [koblas]'s initialization, not from [platformLinearAlgebra], because the two halves are
 * registered separately: a host with CBLAS and no LAPACKE keeps the native level-3 routines and the
 * portable factorizations. A ServiceLoader provider with a higher priority still wins on either half.
 */
internal actual fun registerPlatformBackends() {
    if (!HostBlasCalls.blasAvailable) return
    val blas = HostBlas()
    if (!probe(blas)) return
    registerBlas(blas)
    if (HostBlasCalls.lapackAvailable) registerLapack(HostLapack())
}

/** Instantiate all registered providers, dropping any whose construction fails. */
private fun loadProviders(): List<LinearAlgebra> {
    val providers = ArrayList<LinearAlgebra>()
    val iterator = ServiceLoader.load(LinearAlgebra::class.java).iterator()
    while (true) {
        @Suppress("TooGenericExceptionCaught") // provider loading can throw Error (ServiceConfigurationError)
        try {
            if (!iterator.hasNext()) break
            providers.add(iterator.next())
        } catch (_: Throwable) {
            break // a malformed registration poisons the iterator; keep what loaded so far
        }
    }
    return providers
}

/** A 1x1 gemv forces the provider's native path to actually load and produce a correct result. */
private fun probe(backend: Blas): Boolean {
    @Suppress("TooGenericExceptionCaught") // native load failures surface as UnsatisfiedLinkError
    return try {
        val y = backend.gemv(DenseMatrix(1, 1, doubleArrayOf(2.0)), doubleArrayOf(3.0))
        y.size == 1 && y[0] == 6.0
    } catch (_: Throwable) {
        false
    }
}
