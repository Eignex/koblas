package com.eignex.koblas.internal.backend

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Which halves of a service-loaded provider a pinned deployment takes. A pin speaks for one half, so the
 * other one keeps whatever the provider offers it rather than being turned away with the dense half.
 */
class BackendPinTest {

    @Test
    fun `an unpinned deployment takes every half a provider carries`() {
        assertEquals(BackendSlot.entries.toSet(), offeredHalves("umfpack", null, null))
    }

    @Test
    fun `a dense pin naming another backend leaves the sparse halves`() {
        assertEquals(BackendSlot.sparseHalves, offeredHalves("umfpack", "openblas", null))
    }

    @Test
    fun `a sparse pin naming another backend leaves the dense halves`() {
        assertEquals(BackendSlot.denseHalves, offeredHalves("openblas", null, "umfpack"))
    }

    @Test
    fun `a provider neither pin names is offered nothing`() {
        assertTrue(offeredHalves("superlu", "openblas", "umfpack").isEmpty())
    }

    @Test
    fun `a pin takes the backend it names whole`() {
        assertEquals(BackendSlot.entries.toSet(), offeredHalves("openblas", "openblas", "openblas"))
        // A bundled provider answers to the plain name, as the selection everywhere else does.
        assertEquals(BackendSlot.entries.toSet(), offeredHalves("umfpack-bundled", null, "umfpack"))
    }
}
