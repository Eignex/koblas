package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.hostblas.HostBlas
import com.eignex.koblas.hostblas.HostBlasCalls
import com.eignex.koblas.hostblas.HostLapack
import com.eignex.koblas.registerBackend
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.umfpack.UmfpackCalls
import com.eignex.koblas.umfpack.UmfpackSparseLapack
import java.util.ServiceLoader
import kotlin.math.abs

/**
 * The JVM's backend discovery, run once on the first [com.eignex.koblas.koblas] read.
 *
 * Two sources, in order. koblas's own host-OpenBLAS halves come first when the machine has the library
 * installed, registered separately so a host with CBLAS and no LAPACKE keeps the native level-3 routines
 * and the portable factorizations. Then [LinearAlgebra] providers found via [ServiceLoader], so a
 * third-party backend artifact on the classpath activates with no code changes.
 *
 * Order does not decide the winner — [com.eignex.koblas.registerBackend] ranks by [Backend.priority], so
 * a provider that outranks the built-in one still wins, on either half independently.
 *
 * Each candidate is probed with a tiny computation before being registered, so one whose native library
 * fails to load on the current platform is skipped rather than crashing startup.
 *
 * The `koblas.backend` system property overrides discovery: `reference` registers nothing, leaving the
 * portable [ReferenceLinearAlgebra]; any other value registers only the backend whose [Backend.name]
 * matches. Read once, when [com.eignex.koblas.koblas] initializes.
 */
internal actual fun registerPlatformBackends() {
    val requested = System.getProperty("koblas.backend")
    if (requested == REFERENCE_NAME) return
    registerHostBlas(requested)
    registerUmfpack(requested)
    for (provider in loadProviders().sortedByDescending { it.priority }) {
        if (requested != null && provider.name != requested) continue
        if (!probe(provider)) continue
        // Once, not twice: registerBackend offers the object as every half it implements. The duplicate
        // this replaces was a leftover from collapsing registerBlas/registerLapack into one verb.
        registerBackend(provider)
    }
}

/** koblas's own FFM binding to a host OpenBLAS, when this machine has one. */
private fun registerHostBlas(requested: String?) {
    if (!HostBlasCalls.blasAvailable) return
    val blas = HostBlas()
    if (requested != null && blas.name != requested) return
    if (!probe(blas)) return
    registerBackend(blas)
    if (HostBlasCalls.lapackAvailable) registerBackend(HostLapack())
}

/**
 * koblas's FFM binding to SuiteSparse's UMFPACK, when this machine has it.
 *
 * The sparse counterpart of [registerHostBlas], and separate from it because the two libraries are
 * independent: a machine may have OpenBLAS without SuiteSparse or the reverse, and each half should
 * accelerate on its own.
 */
private fun registerUmfpack(requested: String?) {
    if (!UmfpackCalls.available) return
    val lapack = UmfpackSparseLapack()
    if (requested != null && lapack.name != requested) return
    if (!probeSparse(lapack)) return
    registerBackend(lapack)
}

/**
 * A 2x2 diagonal factor-and-solve forces UMFPACK's native path to load and produce a correct answer.
 *
 * The dense probe multiplies; this one factorizes, because a binding can resolve its symbols and still fail
 * on the first real call — a version whose `di` family differs, or a dependent SuiteSparse library missing.
 */
private fun probeSparse(lapack: SparseLapack): Boolean {
    @Suppress("TooGenericExceptionCaught") // native load failures surface as UnsatisfiedLinkError
    return try {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))
        val f = lapack.factor(a)
        val x = f.solve(doubleArrayOf(2.0, 8.0))
        !f.singular && abs(x[0] - 1.0) < 1e-12 && abs(x[1] - 2.0) < 1e-12
    } catch (_: Throwable) {
        false
    }
}

/** The reserved `koblas.backend` value that means "register nothing". */
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
            break // a malformed registration poisons the iterator; keep what loaded so far
        }
    }
    return providers
}

/** A 1x1 gemv forces the candidate's native path to actually load and produce a correct result. */
private fun probe(backend: Blas): Boolean {
    @Suppress("TooGenericExceptionCaught") // native load failures surface as UnsatisfiedLinkError
    return try {
        val y = backend.gemv(DenseMatrix(1, 1, doubleArrayOf(2.0)), doubleArrayOf(3.0))
        y.size == 1 && y[0] == 6.0
    } catch (_: Throwable) {
        false
    }
}
