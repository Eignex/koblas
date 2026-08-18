package com.eignex.koblas

import com.eignex.koblas.dense.F64PlatformVectorKernels
import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.dense.registerPlatformBackends

/** Where the double-precision registrations live; an element type added later gets a registry beside it. */
private val f64 = F64Registry()

/** Runs platform discovery exactly once, on the first [koblas] read. */
private val discovery: Unit by lazy { registerPlatformBackends() }

/**
 * The process-wide default context: an [installBackends] override when set, else whatever registered
 * itself, else the portable reference implementations. Every free function in koblas uses this.
 */
public val koblas: F64Context
    get() {
        discovery
        return f64.active
    }

/** What this runtime resolved, for startup logging (e.g. `"backend=openblas, kernels=simd(8 lanes)"`). */
public val koblasInfo: String get() = "backend=${koblas.name}, kernels=${koblas.vectorKernels.name}"

/**
 * Offers [backend] for automatic selection as every half it implements. It becomes active only while no
 * [installBackends] override is set and nothing stronger was offered for the same half.
 *
 * A backend is offered to the registry of every element type, since which one it serves is which halves it
 * implements, and one object may serve more than one.
 *
 * @throws IllegalArgumentException if [backend] implements no half of any element type, which would
 *   otherwise register nothing and look like it worked.
 */
public fun registerBackend(backend: Backend) {
    val offered = f64.offer(backend)
    require(offered) {
        "${backend.name} implements none of ${F64Registry.HALF_NAMES}, so there is nothing to register it as"
    }
}

/**
 * Overrides the context [koblas] returns; null restores automatic selection. Not synchronized with
 * operations in flight, so install during startup, before other threads run.
 */
public fun installBackends(context: F64Context?) {
    f64.install(context)
}

/** Test hook: clears every override and registration, so selection tests are order-independent. */
internal fun resetBackends() {
    f64.reset()
}

/** The kernels the compiled-in path uses when nothing is registered, for tests that need to name them. */
internal val platformKernels: F64VectorKernels get() = F64PlatformVectorKernels
