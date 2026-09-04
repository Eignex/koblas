package com.eignex.koblas.internal.backend

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Which halves of a service-loaded provider a role-pinned deployment takes. A pin speaks for one half, so
 * every other role keeps whatever the provider offers it.
 */
class BackendPinTest {

    @Test
    fun `an unpinned deployment takes every half a provider carries`() {
        assertEquals(BackendSlot.entries.toSet(), offerFor("umfpack", unpinned()).halves)
    }

    @Test
    fun `a role pin naming another backend leaves every other role`() {
        val requested = unpinned() + (BackendSlot.F64Blas to "openblas")
        assertEquals(BackendSlot.entries.toSet() - BackendSlot.F64Blas, offerFor("umfpack", requested).halves)
    }

    @Test
    fun `a role pin takes the backend it names`() {
        val requested = BackendSlot.entries.associateWith { "reference" } + (BackendSlot.F64SparseBlas to "umfpack")
        assertEquals(setOf(BackendSlot.F64SparseBlas), offerFor("umfpack", requested).halves)
    }

    @Test
    fun `a provider no role pin names is offered nothing`() {
        val requested = BackendSlot.entries.associateWith { "openblas" }
        assertTrue(offerFor("superlu", requested).isEmpty)
    }

    @Test
    fun `a bundled provider answers to its canonical name`() {
        val requested = BackendSlot.entries.associateWith { "umfpack" }
        // A bundled provider answers to the plain name, as the selection everywhere else does.
        assertEquals(BackendSlot.entries.toSet(), offerFor("umfpack-bundled", requested).halves)
    }

    private fun unpinned(): Map<BackendSlot, String?> = BackendSlot.entries.associateWith { null }
}
