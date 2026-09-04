package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.BackendRole
import com.eignex.koblas.F64Context
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64PlatformKernels

internal object BackendRegistry {

    /** Where the double-precision registrations live; an element type added later gets a registry beside it. */
    private val f64 = F64Registry()

    /** The gate discovery runs behind, which is also what makes a second caller wait for it. */
    private val discovery = RunOnce()

    /**
     * Runs platform discovery once when the application explicitly asks for it, and does not return before
     * it has finished: a caller on another thread waits rather than reading a registry still filling up. A
     * provider that asks for discovery while it is being probed is already inside the pass and returns
     * without starting a second one. A pass that throws leaves whatever it registered and can be retried.
     */
    internal fun discover() {
        discovery.run(::registerPlatformBackends)
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

    /**
     * The backend registered for [slot] under [name], or null when nothing did. Typed by the caller from
     * the same slot, since a seam holds only backends its half accepted.
     */
    internal fun named(slot: BackendSlot, name: String) = f64.named(slot, name)

    /** The names registered for [slot], strongest first. */
    internal fun namesFor(slot: BackendSlot) = f64.namesFor(slot)

    /** The names registered for [role], strongest first. */
    internal fun namesFor(role: BackendRole) = f64.namesFor(role)

    /**
     * Offers one automatically discovered backend without recursively starting discovery. [offered] narrows
     * the offer to the halves a deployment left this backend, for one that pinned another half elsewhere and
     * said nothing about the rest of what this one carries, and says which halves it asked for by name.
     */
    internal fun registerAutomatic(backend: Backend, offered: BackendOffer? = null) {
        offer(backend, explicit = false, offered = offered)
    }

    private fun offer(backend: Backend, explicit: Boolean, offered: BackendOffer? = null) {
        val taken = f64.offer(backend, explicit, offered)
        // A narrowed offer matching nothing is the caller's own doing, not a backend with no half at all.
        require(taken || offered != null) {
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
    internal val platformKernels: F64Kernels get() = F64PlatformKernels
}
