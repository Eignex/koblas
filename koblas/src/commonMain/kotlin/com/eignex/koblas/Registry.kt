package com.eignex.koblas

import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.internal.backend.BackendRegistry

/**
 * The process-wide default context: an [installBackends] override when set, else registered backends, else
 * the portable reference implementations. Every free function in koblas uses this.
 */
public val koblas: F64Context get() = BackendRegistry.activeContext

/** What this runtime resolved, for startup logging (e.g. `"backend=openblas, kernels=simd(8 lanes)"`). */
public val koblasInfo: String get() = "backend=${koblas.name}, kernels=${koblas.vectorKernels.name}"

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

internal val platformKernels: F64VectorKernels get() = BackendRegistry.platformKernels
