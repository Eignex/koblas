package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend

/*
 * What discovery offers a backend it found, which is the same question on every target: the deployment may
 * have pinned some halves to a backend by name, and this one takes whatever is left for it.
 */

/**
 * The halves discovery offers one backend, and which of them the deployment named that backend for.
 *
 * The two are different requests. A half nobody named is one this backend may fill if it is the right kind
 * of provider for it, so the specialization policy applies. A [named] half is a deployment asking for this
 * backend there, which is the same thing [com.eignex.koblas.F64ContextBuilder.withBackend] expresses by
 * naming a role, and it is answered by the type test alone.
 */
internal class BackendOffer(val halves: Set<BackendSlot>, val named: Set<BackendSlot>) {
    init {
        require(named.all { it in halves }) { "a named half has to be one of the offered halves" }
    }

    /** Whether pins left this backend nothing, which is not an offer worth making. */
    val isEmpty: Boolean get() = halves.isEmpty()

    /** Whether the deployment named this backend for [slot], rather than merely leaving it to it. */
    fun wasNamed(slot: BackendSlot): Boolean = slot in named
}

/** What the provider named [name] may fill under the deployment's independent role pins. */
internal fun offerFor(name: String, requested: Map<BackendSlot, String?>): BackendOffer {
    val halves = mutableSetOf<BackendSlot>()
    val named = mutableSetOf<BackendSlot>()
    for (slot in BackendSlot.entries) {
        val pin = requested.getValue(slot)
        when {
            pin == null -> halves += slot

            matchesRequested(name, pin) -> {
                halves += slot
                named += slot
            }
        }
    }
    return BackendOffer(halves, named)
}

/** Registers [backend]'s halves when its library loaded and the deployment did not pin those elsewhere. */
internal fun registerIfOffered(backend: Backend, requested: Map<BackendSlot, String?>) {
    if (!backend.isAvailable) return
    val offered = offerFor(backend.name, requested)
    if (!offered.isEmpty) BackendRegistry.registerAutomatic(backend, offered)
}
