package com.eignex.koblas

import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.internal.backend.BackendRegistry
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.sparse.F64SparseBlas
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
 * Offers [backend] as an explicit choice for every half it implements. Explicit registrations outrank
 * automatically discovered ones; among explicit registrations, [Backend.priority] selects the winner.
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
 * [koblas] hands out one backend per half, the strongest that registered, which is what a caller wanting
 * the best available should take. This is for a caller that wants a particular library instead. The sparse
 * backends are specialised rather than interchangeable, so which is best is a property of the matrix and
 * not of a ranking: KLU wants a repeated circuit pattern, UMFPACK an unstructured system, BASICLU a basis
 * whose columns are replaced one at a time. A solver running an interior point method beside a simplex may
 * want two of them at once, which is what this answers and [koblas] cannot.
 *
 * Named rather than constructed directly so a caller keeps discovery, the configured library paths, and
 * the bundled builds. What comes back is the binding's own type, a bundled provider included, so the
 * routines that library carries outside this seam are reached by narrowing to it:
 * `(sparseDecompositionsNamed("basiclu") as? BasicluSparseLu)?.factorBasis(b)`.
 */
public fun sparseDecompositionsNamed(name: String): F64SparseDecompositions? =
    BackendRegistry.sparseDecompositionsNamed(name)

/**
 * The sparse matrix half registered under [name], the counterpart of [sparseDecompositionsNamed] for the
 * products. Its own lookup because that half now resolves to a binding rather than always to the portable
 * routines, so which one answered is a question a caller can have.
 */
public fun sparseBlasNamed(name: String): F64SparseBlas? = BackendRegistry.sparseBlasNamed(name)

/**
 * The basis solvers registered under [name], the counterpart of [sparseDecompositionsNamed] for the basis half.
 */
public fun basisSolversNamed(name: String): F64BasisSolvers? = BackendRegistry.basisSolversNamed(name)

/** The backends registered for [slot], strongest first, for a diagnostic naming what a lookup can find. */
public fun registeredBackendNames(slot: BackendSlot): List<String> = BackendRegistry.namesFor(slot)

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
