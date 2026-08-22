package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import kotlin.concurrent.atomics.AtomicReference
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/**
 * One replaceable half of the compute seam, holding the strongest offer made for it. A whole-context
 * override is the public registry's job, not this one's.
 *
 * @param T the interface this half implements.
 * @param onChange run after every change, for a seam whose composed whole needs rebuilding.
 */
@OptIn(ExperimentalAtomicApi::class)
internal class Seam<T : Backend>(private val onChange: () -> Unit = {}) {
    private val slot = AtomicReference<T?>(null)

    /**
     * The strongest registration, or null. Null means use the compiled kernel, since a reference
     * implementation defined in terms of itself would recurse.
     */
    val active: T? get() = slot.load()

    /**
     * Offers [backend], which becomes active only if it outranks the strongest previous offer.
     *
     * Compare-and-set rather than a read and a write, so two threads offering at once cannot leave the
     * weaker one holding the seam. A lost race retries against whatever won it, which either supersedes it
     * or ranks it out.
     */
    fun register(backend: T) {
        while (true) {
            val current = slot.load()
            if (current != null && backend.priority <= current.priority) return
            if (slot.compareAndSet(current, backend)) {
                onChange()
                return
            }
        }
    }

    /** Clears the registration, leaving [active] null. */
    fun reset() {
        slot.store(null)
        onChange()
    }
}
