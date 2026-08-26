package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import kotlin.concurrent.atomics.AtomicReference
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/**
 * One replaceable half of the compute seam. A whole-context override is the public registry's job, not
 * this one's.
 *
 * It keeps every offer rather than only the winning one. [active] still answers with the strongest, which
 * is what a caller taking whatever is best gets, but the sparse halves are not a ladder the way the dense
 * ones are: their backends are specialised rather than interchangeable, and one process can want two of
 * them at once. Keeping the rest is what lets [named] hand a caller the particular library it needs
 * without giving up discovery, configuration, and bundling to construct one itself.
 *
 * @param T the interface this half implements.
 * @param onChange run after every change, for a seam whose composed whole needs rebuilding.
 */
@OptIn(ExperimentalAtomicApi::class)
internal class Seam<T : Backend>(private val onChange: () -> Unit = {}) {
    private class Registration<T : Backend>(val backend: T, val explicit: Boolean)

    private val registrations = AtomicReference<List<Registration<T>>>(emptyList())

    /**
     * The strongest explicit offer or, when none exists, the strongest automatic one. Null means use the
     * compiled kernel, since a reference implementation defined in terms of itself would recurse.
     */
    val active: T? get() = strongest(registrations.load())?.backend

    /** Every offer, strongest first, so a caller can see what a lookup had to choose between. */
    val all: List<T> get() = ranked(registrations.load()).map { it.backend }

    /**
     * The offer registered under [name], strongest first where a bundled provider and a configured one
     * both answer to it. Null when nothing registered under that name.
     */
    fun named(name: String): T? = all.firstOrNull { matchesRequested(it.name, name) }

    /**
     * Records [backend] as an offer for this half. An offer of the same name supersedes an earlier one only
     * if it outranks it, so re-registering something weaker leaves the stronger offer standing and the list
     * holds one entry per name rather than growing with every call.
     *
     * Copy-and-set rather than a read and a write, so two threads offering at once cannot lose one of the
     * offers. A lost race retries against whatever won it. Registrations happen at startup and are few, so
     * copying the list costs nothing worth avoiding.
     */
    fun register(backend: T, explicit: Boolean) {
        while (true) {
            val current = registrations.load()
            val incoming = Registration(backend, explicit)
            val existing = current.firstOrNull { it.backend.name == backend.name }
            val next = when {
                existing == null -> current + incoming
                outranks(incoming, existing) -> current.map { if (it === existing) incoming else it }
                else -> return
            }
            if (registrations.compareAndSet(current, next)) {
                onChange()
                return
            }
        }
    }

    /** Clears every registration, leaving [active] null. */
    fun reset() {
        registrations.store(emptyList())
        onChange()
    }

    /**
     * Explicit offers outrank automatic ones, and priority ranks offers of the same kind. Ties keep
     * registration order, so the first to offer holds the half against a later equal.
     */
    private fun ranked(offers: List<Registration<T>>): List<Registration<T>> {
        val explicit = offers.filter { it.explicit }
        val pool = explicit.ifEmpty { offers }
        return pool.sortedByDescending { it.backend.priority }
    }

    private fun strongest(offers: List<Registration<T>>): Registration<T>? = ranked(offers).firstOrNull()

    /** Explicit beats automatic whatever the priorities; otherwise the stronger priority wins. */
    private fun outranks(offer: Registration<T>, standing: Registration<T>): Boolean =
        if (offer.explicit != standing.explicit) offer.explicit else offer.backend.priority > standing.backend.priority
}
