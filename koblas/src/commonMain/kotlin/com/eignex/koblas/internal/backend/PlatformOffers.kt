package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend

/*
 * What discovery offers a backend it found, which is the same question on every target: the deployment may
 * have pinned some halves to a backend by name, and this one takes whatever is left for it.
 */

/** The halves the provider named [name] may fill under the deployment's independent role pins. */
internal fun offeredHalves(name: String, requested: Map<BackendSlot, String?>): Set<BackendSlot> =
    BackendSlot.entries.filterTo(mutableSetOf()) { slot ->
        requested.getValue(slot)?.let { matchesRequested(name, it) } ?: true
    }

/** Registers [backend]'s halves when its library loaded and the deployment did not pin those elsewhere. */
internal fun registerIfOffered(backend: Backend, requested: Map<BackendSlot, String?>) {
    if (!backend.isAvailable) return
    val offered = offeredHalves(backend.name, requested)
    if (offered.isNotEmpty()) BackendRegistry.registerAutomatic(backend, offered)
}
