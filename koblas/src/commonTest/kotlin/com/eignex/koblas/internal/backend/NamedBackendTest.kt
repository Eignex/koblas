package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseDecompositions
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.basis.F64BasisSolvers
import kotlin.test.*

/**
 * The sparse halves hand out one winner, and a caller that wants a particular library asks for it by name.
 * Two specialised backends in one process is the case these are about: a solver running an interior point
 * method beside a simplex wants both, and a ranking can only answer with one.
 */
class NamedBackendTest {

    private class FakeSparseLu(override val name: String, override val priority: Int) :
        F64SparseDecompositions by F64ReferenceSparseLinearAlgebra {
        override fun factor(a: F64SparseMatrix): F64SparseLuFactorization = F64ReferenceSparseLinearAlgebra.factor(a)
    }

    private class FakeBasisSolvers(override val name: String, override val priority: Int) : F64BasisSolvers {
        override fun basisSolver(a: F64SparseMatrix) = F64ReferenceSparseLinearAlgebra.basisSolver(a)
    }

    @Test
    fun `the weaker registration stays reachable by name`() = withCleanBackends {
        registerBackend(FakeSparseLu("weaker", priority = 10))
        registerBackend(FakeSparseLu("stronger", priority = 20))

        assertEquals("stronger", koblas.sparseDecompositions.name, "the half still goes to the strongest")
        assertEquals("weaker", sparseDecompositionsNamed("weaker")?.name, "and the other is still there to ask for")
    }

    @Test
    fun `both specialised backends are usable at once`() = withCleanBackends {
        registerBackend(FakeSparseLu("basis-shaped", priority = 20))
        registerBackend(FakeSparseLu("pattern-shaped", priority = 10))

        val forBases = sparseDecompositionsNamed("basis-shaped")
        val forPatterns = sparseDecompositionsNamed("pattern-shaped")

        assertNotNull(forBases)
        assertNotNull(forPatterns)
        assertNotSame(forBases, forPatterns, "one solver holding two libraries needs two objects")
    }

    @Test
    fun `an unregistered name resolves to nothing`() = withCleanBackends {
        registerBackend(FakeSparseLu("present", priority = 10))

        assertNull(sparseDecompositionsNamed("absent"))
    }

    @Test
    fun `a basis solver is reachable by name`() = withCleanBackends {
        registerBackend(FakeBasisSolvers("basis", priority = 10))

        assertEquals("basis", basisSolversNamed("basis")?.name)
        assertNull(basisSolversNamed("absent"))
    }

    @Test
    fun `platform discovery leaves a usable context`() = withCleanBackends {
        discoverBackends()

        assertTrue(koblas.isAvailable)
    }

    /** Bundled providers add a suffix and still answer to the name a deployment configures. */
    @Test
    fun `a bundled provider answers to its plain name`() = withCleanBackends {
        registerBackend(FakeSparseLu("umfpack-bundled", priority = 10))

        assertEquals("umfpack-bundled", sparseDecompositionsNamed("umfpack")?.name)
    }

    @Test
    fun `the registered names are listed strongest first`() = withCleanBackends {
        registerBackend(FakeSparseLu("weaker", priority = 10))
        registerBackend(FakeSparseLu("stronger", priority = 20))

        assertEquals(listOf("stronger", "weaker"), registeredBackendNames(BackendSlot.F64SparseDecompositions))
    }

    @Test
    fun `re-registering a name replaces that offer rather than adding another`() = withCleanBackends {
        registerBackend(FakeSparseLu("same", priority = 10))
        registerBackend(FakeSparseLu("same", priority = 30))

        assertEquals(listOf("same"), registeredBackendNames(BackendSlot.F64SparseDecompositions))
        assertEquals(30, sparseDecompositionsNamed("same")?.priority)
    }

    /** Keeping one entry per name must not let a later weaker offer of that name take the half. */
    @Test
    fun `re-registering a name weaker leaves the stronger offer standing`() = withCleanBackends {
        registerBackend(FakeSparseLu("same", priority = 30))
        registerBackend(FakeSparseLu("same", priority = 10))

        assertEquals(30, sparseDecompositionsNamed("same")?.priority)
        assertEquals(30, koblas.sparseDecompositions.priority)
    }

    @Test
    fun `keeping every offer leaves the strongest holding the half`() = withCleanBackends {
        registerBackend(FakeSparseLu("stronger", priority = 20))
        registerBackend(FakeSparseLu("weaker", priority = 10))

        assertEquals("stronger", koblas.sparseDecompositions.name, "a later weaker offer must not take the half")
    }

    @Test
    fun `a reset clears what a name could find`() = withCleanBackends {
        registerBackend(FakeSparseLu("present", priority = 10))
        resetBackends()

        assertNull(sparseDecompositionsNamed("present"))
        assertEquals(emptyList(), registeredBackendNames(BackendSlot.F64SparseDecompositions))
    }

    @Test
    fun `the portable fallback is not a registration a name can find`() = withCleanBackends {
        assertEquals(BackendNames.REFERENCE, koblas.sparseDecompositions.name)
        assertNull(sparseDecompositionsNamed(BackendNames.REFERENCE), "nothing registered it; it is the fallback")
    }

    /** An explicit offer takes the half, which is what it is for, and takes nothing else. */
    @Test
    fun `a discovered backend stays reachable while an explicit one holds the half`() = withCleanBackends {
        BackendRegistry.registerAutomatic(FakeSparseLu("discovered", priority = 30))
        registerBackend(FakeSparseLu("configured", priority = 10))

        assertEquals("configured", koblas.sparseDecompositions.name, "the explicit offer holds the half")
        val discovered = sparseDecompositionsNamed("discovered")
        assertEquals("discovered", discovered?.name, "and the discovered one is still there")
    }

    @Test
    fun `a discovered backend is listed behind the explicit one rather than dropped`() = withCleanBackends {
        BackendRegistry.registerAutomatic(FakeSparseLu("discovered", priority = 30))
        registerBackend(FakeSparseLu("configured", priority = 10))

        assertEquals(listOf("configured", "discovered"), registeredBackendNames(BackendSlot.F64SparseDecompositions))
    }

    /** Order of arrival must not change either answer. */
    @Test
    fun `an explicit offer registered first still leaves a later discovered one findable`() = withCleanBackends {
        registerBackend(FakeSparseLu("configured", priority = 10))
        BackendRegistry.registerAutomatic(FakeSparseLu("discovered", priority = 30))

        assertEquals("configured", koblas.sparseDecompositions.name)
        assertEquals("discovered", sparseDecompositionsNamed("discovered")?.name)
    }
}
