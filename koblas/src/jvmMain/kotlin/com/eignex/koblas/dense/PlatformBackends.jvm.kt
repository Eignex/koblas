package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.BackendNames
import com.eignex.koblas.ConfigurationKeys
import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.f64DispatchThresholds
import com.eignex.koblas.hostblas.HostBlas
import com.eignex.koblas.hostblas.HostBlasCalls
import com.eignex.koblas.hostblas.HostLapack
import com.eignex.koblas.registerBackend
import com.eignex.koblas.umfpack.UmfpackCalls
import com.eignex.koblas.umfpack.UmfpackSparseLapack
import java.util.ServiceLoader

/**
 * The JVM's backend discovery, run once on the first [com.eignex.koblas.koblas] read. koblas's own
 * halves first, then [ServiceLoader] providers; [Backend.priority] picks the winner on each half.
 */
internal actual fun registerPlatformBackends() {
    val requested = System.getProperty(ConfigurationKeys.BACKEND_PROPERTY)
    if (requested == BackendNames.REFERENCE) return
    registerHostBlas(requested)
    registerUmfpack(requested)
    for (provider in loadProviders().sortedByDescending { it.priority }) {
        if (requested != null && provider.name != requested) continue
        if (!probe(provider)) continue
        // Once, not per half, since registerBackend offers the object as every half it implements.
        registerBackend(provider)
    }
}

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
 * as [registerHostBlas]. Registration means installed, not working; [UmfpackSparseLapack] falls back.
 */
private fun registerUmfpack(requested: String?) {
    if (!UmfpackCalls.libraryPresent) return
    val lapack = UmfpackSparseLapack()
    if (requested != null && lapack.name != requested) return
    registerBackend(lapack)
}

/** Instantiate all registered providers, dropping any whose construction fails. */
private fun loadProviders(): List<F64LinearAlgebra> {
    val providers = ArrayList<F64LinearAlgebra>()
    val iterator = ServiceLoader.load(F64LinearAlgebra::class.java).iterator()
    while (true) {
        @Suppress("TooGenericExceptionCaught") // provider loading can throw Error (ServiceConfigurationError)
        try {
            if (!iterator.hasNext()) break
            providers.add(iterator.next())
        } catch (_: Throwable) {
            break // a malformed registration poisons the iterator, keep what loaded so far
        }
    }
    return providers
}

/**
 * Makes the candidate load its native path and produce a correct result.
 *
 * Sized from the level-3 gate, because a candidate that gates on these same thresholds answers anything
 * below its own gate from its portable fallback, where a missing library cannot show up. A 1x1 gemv, which
 * this used to be, can never reach a native path: every gate sits above 1, and level 2 stays portable
 * outright on a JVM carrying the Vector API. A level pinned portable has no native path to reach at any
 * size, so the probe stays at the cheap sanity check rather than allocating for the pinning value.
 *
 * The primitive is called rather than the convenience overload, since a backend built by delegation
 * inherits a forwarder for every routine it does not override and the convenience one would bypass it.
 */
internal fun probe(backend: F64Blas): Boolean {
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
