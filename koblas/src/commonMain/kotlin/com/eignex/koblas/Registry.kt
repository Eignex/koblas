package com.eignex.koblas

import com.eignex.koblas.dense.F64PlatformVectorKernels
import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.dense.registerPlatformBackends

/** Where the double-precision registrations live; an element type added later gets a registry beside it. */
private val f64 = F64Registry()

/**
 * Set when discovery starts rather than when it finishes, so a read of [koblas] from inside it does not
 * start it again. Discovery probes each candidate by calling its own `gemv`, and a candidate that has not
 * overridden its vector kernels resolves them through [koblas], which lands back here. The lazy below has
 * not published a value yet at that point, so it would run the initializer a second time: load the
 * providers again, build a fresh instance of each, probe that one, and so on until the stack ran out.
 */
private var discoveryStarted = false

/** Runs platform discovery exactly once, on the first [koblas] read. */
private val discovery: Unit by lazy {
    // Ordered by the lazy's own lock: the nested read is on the thread already inside it, and any other
    // thread waits on that lock rather than reaching this.
    if (!discoveryStarted) {
        discoveryStarted = true
        registerPlatformBackends()
    }
}

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
 * Each half keeps the strongest offer through a read, a compare and a write that are not one atomic step,
 * so concurrent registrations can leave the weaker one active. Register during startup, before other
 * threads run, as [installBackends] also asks.
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
