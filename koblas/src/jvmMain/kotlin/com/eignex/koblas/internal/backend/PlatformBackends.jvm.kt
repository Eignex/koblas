package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.host.jvm.HostBlasCalls
import com.eignex.koblas.hostblas.HostBlas
import com.eignex.koblas.hostblas.HostLapack
import com.eignex.koblas.registerBackend
import com.eignex.koblas.sparse.F64SparseLu
import com.eignex.koblas.sparse.host.superlu.SuperluCalls
import com.eignex.koblas.sparse.host.superlu.SuperluSparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackCalls
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu
import java.util.ServiceLoader

/**
 * The JVM's backend discovery, run once on the first [com.eignex.koblas.koblas] read. koblas's own
 * halves first, then [ServiceLoader] providers; [Backend.priority] picks the winner on each half.
 */
internal actual fun registerPlatformBackends() {
    val denseRequested = System.getProperty(ConfigurationKeys.DENSE_BACKEND_PROPERTY)
    val sparseRequested = System.getProperty(ConfigurationKeys.SPARSE_BACKEND_PROPERTY)
    for (provider in loadProviders().sortedByDescending { it.priority }) {
        if (provider is F64Blas && denseRequested != null &&
            !matchesRequested(provider.name, denseRequested)
        ) {
            continue
        }
        if (provider is F64SparseLu && sparseRequested != null &&
            !matchesRequested(provider.name, sparseRequested)
        ) {
            continue
        }
        if (!probe(provider)) continue
        // Once, not per half, since registerBackend offers the object as every half it implements.
        registerBackend(provider)
    }
    registerHostBlas(denseRequested)
    registerSuperlu(sparseRequested)
    registerUmfpack(sparseRequested)
}

/** Bundled providers add a diagnostic suffix while retaining the canonical name callers configure. */
private fun matchesRequested(name: String, requested: String): Boolean =
    name == requested || name.removeSuffix("-bundled") == requested

/**
 * koblas's own FFM binding to a host OpenBLAS. The test is a `dlopen` plus a symbol lookup, not a probe
 * computation, because `Linker.downcallHandle` is stack-hungry enough to throw StackOverflowError here.
 */
private fun registerHostBlas(requested: String?) {
    if (!HostBlasCalls.available) return
    val blas = HostBlas()
    if (requested != null && blas.name != requested) return
    registerBackend(blas)
    if (HostBlasCalls.lapackAvailable) registerBackend(HostLapack())
}

/**
 * koblas's FFM binding to SuiteSparse's UMFPACK, checked with a bare `dlopen` for the same stack reason
 * as [registerHostBlas]. Registration means installed, not working; [UmfpackSparseLu] falls back.
 */
private fun registerUmfpack(requested: String?) {
    if (!UmfpackCalls.libraryPresent) return
    val lapack = UmfpackSparseLu()
    if (requested != null && lapack.name != requested) return
    registerBackend(lapack)
}

/** SuperLU is the preferred general sparse-LU backend; UMFPACK remains selectable by name. */
private fun registerSuperlu(requested: String?) {
    if (!SuperluCalls.libraryPresent) return
    val lapack = SuperluSparseLu()
    if (requested != null && lapack.name != requested) return
    registerBackend(lapack)
}

/** Instantiate all registered providers, dropping any whose construction fails. */
private fun loadProviders(): List<Backend> {
    val providers = ArrayList<Backend>()
    loadProviders(Backend::class.java, providers)
    // The dense service type was the original public SPI. Keep it while providers migrate to [Backend],
    // which also permits sparse-only add-ons such as UMFPACK.
    loadProviders(com.eignex.koblas.dense.F64LinearAlgebra::class.java, providers)
    return providers.distinctBy { it::class.java.name }
}

/** Adds providers from one service type, dropping a malformed registration without failing discovery. */
private fun <T : Backend> loadProviders(type: Class<T>, providers: MutableList<Backend>) {
    val iterator = ServiceLoader.load(type).iterator()
    while (true) {
        @Suppress("TooGenericExceptionCaught") // provider loading can throw Error (ServiceConfigurationError)
        try {
            if (!iterator.hasNext()) break
            providers.add(iterator.next())
        } catch (_: Throwable) {
            break // a malformed registration poisons the iterator, keep what loaded so far
        }
    }
}

/**
 * Makes the candidate load its native path and produce a correct result.
 *
 * Sized from the level-3 gate, because a candidate that gates on these same thresholds answers anything
 * below its own gate from its portable fallback, where a missing library cannot show up. Every gate sits
 * above 1, and level 2 stays portable outright on a JVM carrying the Vector API. A level pinned portable
 * has no native path to reach at any size, so the probe stays at the cheap sanity check rather than
 * allocating for the pinning value.
 *
 * The primitive is called rather than the convenience overload, since a backend built by delegation
 * inherits a forwarder for every routine it does not override and the convenience one would bypass it.
 */
internal fun probe(backend: Backend): Boolean {
    if (backend !is F64Blas) return backend.isAvailable
    val gate = f64DispatchThresholds.level3
    val n = if (gate == Int.MAX_VALUE) 1 else maxOf(1, gate)
    @Suppress("TooGenericExceptionCaught") // native load failures surface as UnsatisfiedLinkError
    return try {
        val a = F64DenseMatrix(n, n, DoubleArray(n * n) { 2.0 })
        val c = F64DenseMatrix(n, n)
        backend.gemm(1.0, a, transposeA = false, a, transposeB = false, beta = 0.0, c = c)
        val expected = 4.0 * n
        c.data.all { it == expected }
    } catch (_: Throwable) {
        false
    }
}
