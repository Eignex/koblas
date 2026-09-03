package com.eignex.koblas

import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.internal.backend.BackendRegistry
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
 * The basis solvers registered under [name], looked up the way the semantic capabilities are.
 *
 * A door of its own for one of the capabilities, which [backendNamed] covers for all of them. Kept for one
 * release so a caller pinning a basis solver by name has somewhere to move to.
 */
@Deprecated(
    "One capability's own lookup; backendNamed covers every capability.",
    ReplaceWith("backendNamed(name, F64Capabilities.basisSolvers)"),
)
public fun basisSolversNamed(name: String): F64BasisSolvers? = backendNamed(name, F64Capabilities.basisSolvers)

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
