package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.F64Context
import com.eignex.koblas.dense.F64PlatformVectorKernels
import com.eignex.koblas.dense.F64VectorKernels
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi

@OptIn(ExperimentalAtomicApi::class)
internal object BackendRegistry {

    /** Where the double-precision registrations live; an element type added later gets a registry beside it. */
    private val f64 = F64Registry()

    /**
     * Set when discovery starts rather than when it finishes, so an explicit discovery request from a provider
     * cannot start it again while its candidate is being probed.
     */
    private val discoveryStarted = AtomicInt(0)

    /** Runs platform discovery once when the application explicitly asks for it. */
    internal fun discover() {
        if (discoveryStarted.compareAndSet(0, 1)) registerPlatformBackends()
    }

    /**
     * The process-wide default context: an explicit override when set, else whatever registered
     * itself, else the portable reference implementations. Every free function in koblas uses this.
     */
    internal val activeContext: F64Context
        get() {
            return f64.active
        }

    /**
     * Offers [backend] explicitly as every half it implements. Explicit offers outrank automatic ones;
     * priority ranks offers of the same kind.
     *
     * A backend is offered to the registry of every element type, since which one it serves is which halves it
     * implements, and one object may serve more than one.
     *
     * Safe to call from any thread: each half takes its offer through a compare-and-set, so concurrent
     * registrations settle on the strongest explicit offer rather than on whoever wrote last. Whole-context
     * installation still wants startup, since it replaces the whole context rather than ranking into a half.
     *
     * @throws IllegalArgumentException if [backend] implements no half of any element type, which would
     *   otherwise register nothing and look like it worked.
     */
    internal fun register(backend: Backend) {
        offer(backend, explicit = true)
    }

    /** Offers one automatically discovered backend without recursively starting discovery. */
    internal fun registerAutomatic(backend: Backend) {
        offer(backend, explicit = false)
    }

    private fun offer(backend: Backend, explicit: Boolean) {
        val offered = f64.offer(backend, explicit)
        require(offered) {
            "${backend.name} implements none of ${BackendSlot.names}, so there is nothing to register it as"
        }
    }

    /**
     * Overrides the process-wide context; null restores registrations. Not synchronized with
     * operations in flight, so install during startup, before other threads run.
     */
    internal fun install(context: F64Context?) {
        f64.install(context)
    }

/**
     * Test hook: clears every override and registration, leaving the portable fallbacks. Platform discovery is
     * not replayed by this, so pair it with rediscovery to put the process back as it was.
     */
    internal fun reset() {
        f64.reset()
    }

    /**
     * Test hook: runs platform discovery again after [reset] cleared the discovered registrations.
     */
    internal fun rediscover() {
        registerPlatformBackends()
    }

    /** The kernels the compiled-in path uses when nothing is registered, for tests that need to name them. */
    internal val platformKernels: F64VectorKernels get() = F64PlatformVectorKernels
}
