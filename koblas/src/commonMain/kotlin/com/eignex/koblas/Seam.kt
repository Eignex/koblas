package com.eignex.koblas

import kotlin.concurrent.Volatile

/**
 * One replaceable half of the compute seam, holding the strongest offer made for it. A whole-context
 * override is [installBackends]'s job, not this one's.
 *
 * @param T the interface this half implements.
 * @param onChange run after every change, for a seam whose composed whole needs rebuilding.
 */
internal class Seam<T : Backend>(private val onChange: () -> Unit = {}) {
    /**
     * The strongest registration, or null. Null means use the compiled kernel, since a reference
     * implementation defined in terms of itself would recurse.
     */
    @Volatile
    var active: T? = null
        private set

    /** Offers [backend], which becomes active only if it outranks the strongest previous offer. */
    fun register(backend: T) {
        val current = active
        if (current == null || backend.priority > current.priority) {
            active = backend
            onChange()
        }
    }

    /** Clears the registration, leaving [active] null. */
    fun reset() {
        active = null
        onChange()
    }
}
