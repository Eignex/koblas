package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseBlas
import kotlin.test.*

class BackendSelectionTest {

    private class FakeBlas(override val name: String, override val priority: Int) : F64Blas by F64ReferenceBlas

    private class NotABackend(override val name: String = "nothing") : Backend

    /** A provider carrying a dense half and a sparse one, as an add-on binding a whole library can. */
    private class FakeBoth(override val name: String, override val priority: Int) :
        F64Blas by F64ReferenceBlas,
        F64SparseBlas by F64ReferenceSparseLinearAlgebra {
        override val isAvailable: Boolean get() = true
        override val isPortable: Boolean get() = false
        override val unavailableReason: String? get() = null
    }

    @Test
    fun `an empty registry resolves to the reference backend`() {
        withCleanBackends {
            assertSame(F64ReferenceBlas, koblas.blas)
            assertEquals("reference", koblas.name)
        }
    }

    @Test
    fun `registration keeps the highest priority and install overrides everything`() {
        withCleanBackends {
            registerBackend(FakeBlas("low", 5))
            assertEquals("low+reference", koblas.name)
            registerBackend(FakeBlas("high", 50))
            assertEquals("high+reference", koblas.name)
            registerBackend(FakeBlas("mid", 20))
            assertEquals("high+reference", koblas.name)
            val manual = FakeBlas("manual", -1)
            installBackends(koblas.with(blas = manual))
            assertEquals("manual+reference", koblas.name)
            installBackends(null)
            assertEquals("high+reference", koblas.name)
        }
    }

    @Test
    fun `an explicit registration outranks automatic discovery`() {
        withCleanBackends {
            BackendRegistry.registerAutomatic(FakeBlas("automatic", 100))
            val configured = FakeBlas("configured", 1)
            registerBackend(configured)
            BackendRegistry.registerAutomatic(FakeBlas("later automatic", 200))
            assertSame(configured, koblas.blas)
        }
    }

    @Test
    fun `the incumbent backend survives a cleared registry`() {
        // Restoring the incumbent keeps the host BLAS suites valid whatever order tests run in. Compared by
        // name rather than by identity: the registry is put back by replaying discovery, which builds a
        // fresh binding object, and what has to survive is which backend fills the half. A restore that
        // failed outright would leave the reference here and still be caught.
        val before = koblas.blas.name
        withCleanBackends { assertSame(F64ReferenceBlas, koblas.blas) }
        assertEquals(before, koblas.blas.name)
    }

    /**
     * A deployment pinning one half names a backend for that half and says nothing about the other, so the
     * halves it did not speak for still take what a provider offers them.
     */
    @Test
    fun `an offer narrowed to some halves leaves the rest of the registry alone`() {
        withCleanBackends {
            BackendRegistry.registerAutomatic(
                FakeBoth("both-halves", 100),
                BackendOffer(BackendSlot.sparseHalves, named = emptySet()),
            )

            assertTrue("both-halves" in BackendRegistry.namesFor(BackendSlot.F64SparseBlas), "the sparse half")
            assertFalse("both-halves" in BackendRegistry.namesFor(BackendSlot.F64Blas), "the dense half")
            assertSame(F64ReferenceBlas, koblas.blas)
        }
    }

    @Test
    fun `registering a backend that implements no half fails loudly`() {
        withCleanBackends {
            val failure = assertFailsWith<IllegalArgumentException> { registerBackend(NotABackend()) }
            assertTrue(failure.message!!.contains("nothing"), "the message should name the backend")
        }
    }

    @Test
    fun `the reset hook clears the install override too`() {
        withCleanBackends {
            val manual = FakeBlas("manual", -1)
            installBackends(koblas.with(blas = manual))
            resetBackends()
            assertSame(F64ReferenceBlas, koblas.blas, "reset must clear the override, not just registration")
        }
    }
}
