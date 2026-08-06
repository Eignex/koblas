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
 * A third-party provider is probed with a tiny computation before being registered, so one whose native
 * library fails to load on the current platform is skipped rather than crashing startup. koblas's own two
 * bindings are not: each answers "is my library here" with a `dlopen` and a symbol lookup, which is both
 * cheaper and shallower than a computation — see [registerHostBlas] and [registerUmfpack] for why the depth
 * matters. External code gets the stricter treatment because it is external code, and because a provider's
 * `gemv` has no per-call fallback behind it.
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
        // Once, not per half: registerBackend already offers the object as every half it implements.
        registerBackend(provider)
    }
}

/**
 * koblas's own FFM binding to a host OpenBLAS, when this machine has one.
 *
 * `blasAvailable` is a `dlopen` plus a lookup of `cblas_dgemm`, and that is the whole test. Confirming the
 * library by running a computation would bind a downcall handle inside the discovery window, where
 * `Linker.downcallHandle` is stack-hungry enough to throw `StackOverflowError`. A resolved symbol is evidence
 * enough that the library is the one it claims to be.
 */
private fun registerHostBlas(requested: String?) {
    if (!HostBlasCalls.blasAvailable) return
    val blas = HostBlas()
    if (requested != null && blas.name != requested) return
    registerBackend(blas)
    if (HostBlasCalls.lapackAvailable) registerBackend(HostLapack())
}

/**
 * koblas's FFM binding to SuiteSparse's UMFPACK, when this machine has it.
 *
 * The sparse counterpart of [registerHostBlas], and separate from it because the two libraries are
 * independent: a machine may have OpenBLAS without SuiteSparse or the reverse, and each half should
 * accelerate on its own.
 *
 * Deliberately does the *cheapest possible* check here — a `dlopen`, via `libraryPresent` — and does not bind
 * a single downcall handle or run any probe computation. Discovery runs at whatever stack depth the first
 * `koblas` read happens to sit at, and creating handles there threw `StackOverflowError` in a full test suite
 * while working fine when the same code ran alone; because the failure was caught as `Throwable` it was
 * reported as "SuiteSparse is not installed" and the backend silently never registered. Binding is now the
 * first real call's business, where the stack is shallow. See `UmfpackCalls` for the whole story.
 *
 * The consequence worth knowing: registration now means "the library is installed", not "the library works".
 * If the symbols turn out not to bind, [UmfpackSparseLapack.factor] falls back to the portable factorization
 * exactly as it does for any other UMFPACK failure, so a caller still gets a correct answer.
 */
private fun registerUmfpack(requested: String?) {
    if (!UmfpackCalls.libraryPresent) return
    val lapack = UmfpackSparseLapack()
    if (requested != null && lapack.name != requested) return
    registerBackend(lapack)
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
