package com.eignex.koblas

import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.internal.backend.BackendRegistry
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.sparse.F64SparseDecompositions
import com.eignex.koblas.sparse.basis.F64BasisSolvers

/**
 * The process-wide default context: an [installBackends] override when set, else registered backends, else
 * the portable reference implementations. Every free function in koblas uses this.
 */
public val koblas: F64Context get() = BackendRegistry.activeContext

/** What this runtime resolved, for startup logging (e.g. `"backend=openblas, kernels=simd(8 lanes)"`). */
public val koblasInfo: String get() = "backend=${koblas.name}, kernels=${koblas.kernels.name}"

/**
 * Offers [backend] as an explicit choice for every role it implements. Explicit registrations outrank
 * automatically discovered ones; among providers for the same role, [Backend.priority] selects the winner.
 */
public fun registerBackend(backend: Backend) {
    BackendRegistry.register(backend)
}

/** Runs automatic platform discovery once, registering whatever host backends are available. */
public fun discoverBackends() {
    BackendRegistry.discover()
}

/**
 * The sparse LU registered under [name], such as `"umfpack"` or `"basiclu"`, or null when nothing did. A
 * bundled provider answers to the plain name as well as its own.
 *
 * [koblas] hands out one provider per semantic role. This is for a caller that wants a particular library
 * instead. Sparse backends are specialised rather than interchangeable: KLU wants a repeated circuit
 * pattern, UMFPACK an unstructured system, and BASICLU a basis whose columns are replaced one at a time.
 * One process may need several of them concurrently.
 *
 * Named rather than constructed directly so a caller keeps discovery, the configured library paths, and
 * the bundled builds. Prefer typed [backendNamed] lookup for semantic capabilities such as
 * `backendNamed("basiclu", F64Capabilities.basisFactorizations)`; this legacy function remains for source
 * compatibility and backend-specific APIs.
 */
public fun sparseDecompositionsNamed(name: String): F64SparseDecompositions? =
    BackendRegistry.sparseDecompositionsNamed(name)

/**
 * The basis solvers registered under [name], the counterpart of [sparseDecompositionsNamed] for the basis half.
 */
public fun basisSolversNamed(name: String): F64BasisSolvers? = BackendRegistry.basisSolversNamed(name)

/** The backends registered for [slot], strongest first, for a diagnostic naming what a lookup can find. */
public fun registeredBackendNames(slot: BackendSlot): List<String> = BackendRegistry.namesFor(slot)

/** The backends registered for [role], strongest first. */
public fun registeredBackendNames(role: BackendRole): List<String> = BackendRegistry.namesFor(role)

/** Overrides the context [koblas] returns; null restores automatic selection. */
public fun installBackends(context: F64Context?) {
    BackendRegistry.install(context)
}

internal fun resetBackends() {
    BackendRegistry.reset()
}

internal fun rediscoverBackends() {
    BackendRegistry.rediscover()
}

internal val platformKernels: F64Kernels get() = BackendRegistry.platformKernels
