package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseBlas
import kotlin.test.*

class BackendSelectionTest {

    private class Fake(override val name: String, override val priority: Int) :
        F64LinearAlgebra by F64ReferenceLinearAlgebra

    private class FakeBlas(override val name: String, override val priority: Int) : F64Blas by F64ReferenceLinearAlgebra

    private class FakeLapack(override val name: String, override val priority: Int) :
        F64Decompositions by F64ReferenceLinearAlgebra

    private class NotABackend(override val name: String = "nothing") : Backend

    /** A provider carrying a dense half and a sparse one, as an add-on binding a whole library can. */
    private class FakeBoth(override val name: String, override val priority: Int) :
        F64Blas by F64ReferenceLinearAlgebra,
        F64SparseBlas by F64ReferenceSparseLinearAlgebra {
        override val isAvailable: Boolean get() = true
        override val isPortable: Boolean get() = false
    }

    @Test
    fun `an empty registry resolves to the reference backend`() {
        withCleanBackends {
            assertSame(F64ReferenceLinearAlgebra, koblas.blas)
            assertSame(F64ReferenceLinearAlgebra, koblas.decompositions)
            assertEquals("reference", koblas.name)
        }
    }

    @Test
    fun `registration keeps the highest priority and install overrides everything`() {
        withCleanBackends {
            registerBackend(Fake("low", 5))
            assertEquals("low+reference", koblas.name)
            registerBackend(Fake("high", 50))
            assertEquals("high+reference", koblas.name)
            registerBackend(Fake("mid", 20)) // weaker than the incumbent: ignored
            assertEquals("high+reference", koblas.name)
            val manual = Fake("manual", -1)
            installBackends(koblas.with(blas = manual, decompositions = manual))
            assertEquals("manual+reference", koblas.name)
            installBackends(null)
            assertEquals("high+reference", koblas.name)
        }
    }

    @Test
    fun `an explicit registration outranks automatic discovery`() {
        withCleanBackends {
            BackendRegistry.registerAutomatic(Fake("automatic", 100))
            val configured = Fake("configured", 1)
            registerBackend(configured)
            BackendRegistry.registerAutomatic(Fake("later automatic", 200))
            assertSame(configured, koblas.blas)
            assertSame(configured, koblas.decompositions)
        }
    }

    @Test
    fun `the incumbent backend survives a cleared registry`() {
        // Restoring the incumbent keeps the host BLAS suites valid whatever order tests run in. Compared by
        // name rather than by identity: the registry is put back by replaying discovery, which builds a
        // fresh binding object, and what has to survive is which backend fills the half. A restore that
        // failed outright would leave the reference here and still be caught.
        val before = koblas.blas.name
        withCleanBackends { assertSame(F64ReferenceLinearAlgebra, koblas.blas) }
        assertEquals(before, koblas.blas.name)
    }

    /**
     * A deployment pinning one half names a backend for that half and says nothing about the other, so the
     * halves it did not speak for still take what a provider offers them.
     */
    @Test
    fun `an offer narrowed to some halves leaves the rest of the registry alone`() {
        withCleanBackends {
            BackendRegistry.registerAutomatic(FakeBoth("both-halves", 100), BackendSlot.sparseHalves)

            assertTrue("both-halves" in BackendRegistry.namesFor(BackendSlot.F64SparseBlas), "the sparse half")
            assertFalse("both-halves" in BackendRegistry.namesFor(BackendSlot.F64Blas), "the dense half")
            assertSame(F64ReferenceLinearAlgebra, koblas.blas)
        }
    }

    @Test
    fun `the halves are selected independently`() {
        withCleanBackends {
            registerBackend(FakeBlas("fastblas", 50))
            assertEquals("fastblas+reference", koblas.name)
            assertEquals(2, koblas.factor(F64DenseMatrix.diagonal(2)).n)
            registerBackend(FakeLapack("fastlapack", 10))
            assertEquals("fastblas+fastlapack+reference", koblas.name)
            registerBackend(FakeBlas("slowblas", 5))
            assertEquals("fastblas+fastlapack+reference", koblas.name)
        }
    }

    @Test
    fun `a backend providing both halves is used for both`() {
        withCleanBackends {
            val both = Fake("whole", 30)
            registerBackend(both)
            assertSame(both, koblas.blas)
            assertSame(both, koblas.decompositions)
            assertEquals("whole+reference", koblas.name, "the dense halves are its; the sparse ones are not")
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
            val manual = Fake("manual", -1)
            installBackends(koblas.with(blas = manual, decompositions = manual))
            resetBackends()
            assertSame(F64ReferenceLinearAlgebra, koblas.blas, "reset must clear the override, not just registration")
        }
    }
}
