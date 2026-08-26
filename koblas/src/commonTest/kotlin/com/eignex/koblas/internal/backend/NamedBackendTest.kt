package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseLu
import kotlin.test.*

/**
 * The sparse halves hand out one winner, and a caller that wants a particular library asks for it by name.
 * Two specialised backends in one process is the case these are about: a solver running an interior point
 * method beside a simplex wants both, and a ranking can only answer with one.
 */
class NamedBackendTest {

    private class FakeSparseLu(override val name: String, override val priority: Int) :
        F64SparseLu by F64ReferenceSparseLinearAlgebra {
        override fun factor(a: F64SparseMatrix, equilibrate: Boolean, dropTolerance: Double): F64SparseFactorization =
            F64ReferenceSparseLinearAlgebra.factor(a, equilibrate, dropTolerance)
    }

    @Test
    fun `the weaker registration stays reachable by name`() = withCleanBackends {
        registerBackend(FakeSparseLu("weaker", priority = 10))
        registerBackend(FakeSparseLu("stronger", priority = 20))

        assertEquals("stronger", koblas.sparseLu.name, "the half still goes to the strongest")
        assertEquals("weaker", sparseLuNamed("weaker")?.name, "and the other is still there to ask for")
    }

    @Test
    fun `both specialised backends are usable at once`() = withCleanBackends {
        registerBackend(FakeSparseLu("basis-shaped", priority = 20))
        registerBackend(FakeSparseLu("pattern-shaped", priority = 10))

        val forBases = sparseLuNamed("basis-shaped")
        val forPatterns = sparseLuNamed("pattern-shaped")

        assertNotNull(forBases)
        assertNotNull(forPatterns)
        assertNotSame(forBases, forPatterns, "one solver holding two libraries needs two objects")
    }

    @Test
    fun `an unregistered name resolves to nothing`() = withCleanBackends {
        registerBackend(FakeSparseLu("present", priority = 10))

        assertNull(sparseLuNamed("absent"))
    }

    /** Bundled providers add a suffix and still answer to the name a deployment configures. */
    @Test
    fun `a bundled provider answers to its plain name`() = withCleanBackends {
        registerBackend(FakeSparseLu("umfpack-bundled", priority = 10))

        assertEquals("umfpack-bundled", sparseLuNamed("umfpack")?.name)
    }

    @Test
    fun `the registered names are listed strongest first`() = withCleanBackends {
        registerBackend(FakeSparseLu("weaker", priority = 10))
        registerBackend(FakeSparseLu("stronger", priority = 20))

        assertEquals(listOf("stronger", "weaker"), registeredBackendNames(BackendSlot.F64SparseLu))
    }

    @Test
    fun `re-registering a name replaces that offer rather than adding another`() = withCleanBackends {
        registerBackend(FakeSparseLu("same", priority = 10))
        registerBackend(FakeSparseLu("same", priority = 30))

        assertEquals(listOf("same"), registeredBackendNames(BackendSlot.F64SparseLu))
        assertEquals(30, sparseLuNamed("same")?.priority)
    }

    /** Keeping one entry per name must not let a later weaker offer of that name take the half. */
    @Test
    fun `re-registering a name weaker leaves the stronger offer standing`() = withCleanBackends {
        registerBackend(FakeSparseLu("same", priority = 30))
        registerBackend(FakeSparseLu("same", priority = 10))

        assertEquals(30, sparseLuNamed("same")?.priority)
        assertEquals(30, koblas.sparseLu.priority)
    }

    @Test
    fun `keeping every offer leaves the strongest holding the half`() = withCleanBackends {
        registerBackend(FakeSparseLu("stronger", priority = 20))
        registerBackend(FakeSparseLu("weaker", priority = 10))

        assertEquals("stronger", koblas.sparseLu.name, "a later weaker offer must not take the half")
    }

    @Test
    fun `a reset clears what a name could find`() = withCleanBackends {
        registerBackend(FakeSparseLu("present", priority = 10))
        resetBackends()

        assertNull(sparseLuNamed("present"))
        assertEquals(emptyList(), registeredBackendNames(BackendSlot.F64SparseLu))
    }

    @Test
    fun `the portable fallback is not a registration a name can find`() = withCleanBackends {
        assertEquals(BackendNames.REFERENCE, koblas.sparseLu.name)
        assertNull(sparseLuNamed(BackendNames.REFERENCE), "nothing registered it; it is the fallback")
    }
}
