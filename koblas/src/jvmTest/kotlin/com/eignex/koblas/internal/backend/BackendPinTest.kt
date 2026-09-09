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
        assertEquals(BackendSlot.entries.toSet(), offerFor(namedProvider("vendor"), unpinned()).halves)
    }

    @Test
    fun `a role pin naming another backend leaves every other role`() {
        val requested = unpinned() + (BackendSlot.Blas to "external-blas")
        assertEquals(
            BackendSlot.entries.toSet() - BackendSlot.Blas,
            offerFor(namedProvider("vendor"), requested).halves,
        )
    }

    @Test
    fun `a role pin takes the backend it names`() {
        val requested = BackendSlot.entries.associateWith { "reference" } + (BackendSlot.SparseBlas to "vendor")
        assertEquals(setOf(BackendSlot.SparseBlas), offerFor(namedProvider("vendor"), requested).halves)
    }

    @Test
    fun `a provider no role pin names is offered nothing`() {
        val requested = BackendSlot.entries.associateWith { "external" }
        assertTrue(offerFor(namedProvider("superlu"), requested).isEmpty)
    }

    @Test
    fun `a bundled provider answers to its canonical name`() {
        val requested = BackendSlot.entries.associateWith { "vendor" }
        // A bundled provider answers to the plain name, as the selection everywhere else does.
        assertEquals(
            BackendSlot.entries.toSet(),
            offerFor(bundledProvider("vendor-bundled", "vendor"), requested).halves,
        )
    }

    private fun unpinned(): Map<BackendSlot, String?> = BackendSlot.entries.associateWith { null }
}
