package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.DenseMatrix
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
    val requested = System.getProperty("koblas.backend")
    if (requested == REFERENCE_NAME) return
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

/** The reserved `koblas.backend` value meaning register nothing. */
private const val REFERENCE_NAME = "reference"

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
            break // a malformed registration poisons the iterator, keep what loaded so far
        }
    }
    return providers
}

/** A 1x1 gemv forces the candidate's native path to actually load and produce a correct result. */
internal fun probe(backend: Blas): Boolean {
    @Suppress("TooGenericExceptionCaught") // native load failures surface as UnsatisfiedLinkError
    return try {
        val y = backend.gemv(DenseMatrix(1, 1, doubleArrayOf(2.0)), doubleArrayOf(3.0))
        y.size == 1 && y[0] == 6.0
    } catch (_: Throwable) {
        false
    }
}
