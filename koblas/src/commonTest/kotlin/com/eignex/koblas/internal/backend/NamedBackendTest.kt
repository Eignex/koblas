package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.BundledBackend
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.GeneralSparseLu
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.SparseLapack
import com.eignex.koblas.sparse.SparseLuFactorization
import com.eignex.koblas.sparse.basis.BasisSolvers
import kotlin.test.*

/**
 * The sparse halves hand out one winner, and a caller that wants a particular library asks for it by name.
 * Two specialised backends in one process is the case these are about: a solver running an interior point
 * method beside a simplex wants both, and a ranking can only answer with one.
 */
class NamedBackendTest {

    /** Fills the general-LU role, which is what a backend offers now that the wide seam alone offers nothing. */
    private class FakeSparseLu(override val name: String, override val priority: Int) :
        SparseLapack by ReferenceSparseLinearAlgebra,
        GeneralSparseLu {
        override fun factor(a: SparseMatrix): SparseLuFactorization = ReferenceSparseLinearAlgebra.factor(a)
    }

    /** The same carrying a bundled build of [canonicalName], which is how a deployment configures it. */
    private class FakeBundledSparseLu(
        override val name: String,
        override val canonicalName: String,
        override val priority: Int,
    ) : SparseLapack by ReferenceSparseLinearAlgebra,
        GeneralSparseLu,
        BundledBackend {
        override fun factor(a: SparseMatrix): SparseLuFactorization = ReferenceSparseLinearAlgebra.factor(a)
    }

    private class FakeBasisSolvers(override val name: String, override val priority: Int) : BasisSolvers {
        override fun basisSolver(a: SparseMatrix) = ReferenceSparseLinearAlgebra.basisSolver(a)
    }

    @Test
    fun `the weaker registration stays reachable by name`() = withCleanBackends {
        registerBackend(FakeSparseLu("weaker", priority = 10))
        registerBackend(FakeSparseLu("stronger", priority = 20))

        assertEquals("stronger", koblas.generalSparseLu.name, "the role still goes to the strongest")
        assertEquals(
            "weaker",
            backendNamed("weaker", Capabilities.generalSparseLu)?.name,
            "and the other is still there to ask for",
        )
    }

    @Test
    fun `both specialised backends are usable at once`() = withCleanBackends {
        registerBackend(FakeSparseLu("basis-shaped", priority = 20))
        registerBackend(FakeSparseLu("pattern-shaped", priority = 10))

        val forBases = backendNamed("basis-shaped", Capabilities.generalSparseLu)
        val forPatterns = backendNamed("pattern-shaped", Capabilities.generalSparseLu)

        assertNotNull(forBases)
        assertNotNull(forPatterns)
        assertNotSame(forBases, forPatterns, "one solver holding two libraries needs two objects")
    }

    @Test
    fun `an unregistered name resolves to nothing`() = withCleanBackends {
        registerBackend(FakeSparseLu("present", priority = 10))

        assertNull(backendNamed("absent", Capabilities.generalSparseLu))
    }

    @Test
    fun `a basis solver is reachable by name`() = withCleanBackends {
        registerBackend(FakeBasisSolvers("basis", priority = 10))

        assertEquals("basis", backendNamed("basis", Capabilities.basisSolvers)?.name)
        assertNull(backendNamed("absent", Capabilities.basisSolvers))
    }

    @Test
    fun `platform discovery leaves a usable context`() = withCleanBackends {
        discoverBackends()

        assertTrue(koblas.isAvailable)
    }

    /** A bundled provider answers to the name a deployment configures as well as to its own. */
    @Test
    fun `a bundled provider answers to the name it was configured under`() = withCleanBackends {
        registerBackend(FakeBundledSparseLu("vendor-bundled", canonicalName = "vendor", priority = 10))

        assertEquals("vendor-bundled", backendNamed("vendor", Capabilities.generalSparseLu)?.name)
        assertEquals("vendor-bundled", backendNamed("vendor-bundled", Capabilities.generalSparseLu)?.name)
    }

    /** The suffix is a diagnostic, not a rule: what a provider bundles is something it declares. */
    @Test
    fun `a provider merely named like a bundled one does not answer for it`() = withCleanBackends {
        registerBackend(FakeSparseLu("vendor-bundled", priority = 10))

        assertNull(backendNamed("vendor", Capabilities.generalSparseLu))
    }

    @Test
    fun `the registered names are listed strongest first`() = withCleanBackends {
        registerBackend(FakeSparseLu("weaker", priority = 10))
        registerBackend(FakeSparseLu("stronger", priority = 20))

        assertEquals(listOf("stronger", "weaker"), registeredBackendNames(BackendRole.SPARSE_GENERAL_LU))
    }

    @Test
    fun `re-registering a name replaces that offer rather than adding another`() = withCleanBackends {
        registerBackend(FakeSparseLu("same", priority = 10))
        registerBackend(FakeSparseLu("same", priority = 30))

        assertEquals(listOf("same"), registeredBackendNames(BackendRole.SPARSE_GENERAL_LU))
        assertEquals(30, backendNamed("same", Capabilities.generalSparseLu)?.priority)
    }

    /** Keeping one entry per name must not let a later weaker offer of that name take the half. */
    @Test
    fun `re-registering a name weaker leaves the stronger offer standing`() = withCleanBackends {
        registerBackend(FakeSparseLu("same", priority = 30))
        registerBackend(FakeSparseLu("same", priority = 10))

        assertEquals(30, backendNamed("same", Capabilities.generalSparseLu)?.priority)
        assertEquals(30, koblas.sparseDecompositions.priority)
    }

    @Test
    fun `keeping every offer leaves the strongest holding the half`() = withCleanBackends {
        registerBackend(FakeSparseLu("stronger", priority = 20))
        registerBackend(FakeSparseLu("weaker", priority = 10))

        assertEquals("stronger", koblas.generalSparseLu.name, "a later weaker offer must not take the role")
    }

    @Test
    fun `a reset clears what a name could find`() = withCleanBackends {
        registerBackend(FakeSparseLu("present", priority = 10))
        resetBackends()

        assertNull(backendNamed("present", Capabilities.generalSparseLu))
        assertEquals(emptyList(), registeredBackendNames(BackendRole.SPARSE_GENERAL_LU))
    }

    @Test
    fun `the portable fallback is not a registration a name can find`() = withCleanBackends {
        assertEquals(BackendNames.REFERENCE, koblas.generalSparseLu.name)
        assertNull(
            backendNamed(BackendNames.REFERENCE, Capabilities.generalSparseLu),
            "nothing registered it; it is the fallback",
        )
    }

    /** An explicit offer takes the half, which is what it is for, and takes nothing else. */
    @Test
    fun `a discovered backend stays reachable while an explicit one holds the half`() = withCleanBackends {
        BackendRegistry.registerAutomatic(FakeSparseLu("discovered", priority = 30))
        registerBackend(FakeSparseLu("configured", priority = 10))

        assertEquals("configured", koblas.generalSparseLu.name, "the explicit offer holds the role")
        val discovered = backendNamed("discovered", Capabilities.generalSparseLu)
        assertEquals("discovered", discovered?.name, "and the discovered one is still there")
    }

    @Test
    fun `a discovered backend is listed behind the explicit one rather than dropped`() = withCleanBackends {
        BackendRegistry.registerAutomatic(FakeSparseLu("discovered", priority = 30))
        registerBackend(FakeSparseLu("configured", priority = 10))

        assertEquals(listOf("configured", "discovered"), registeredBackendNames(BackendRole.SPARSE_GENERAL_LU))
    }

    /** Order of arrival must not change either answer. */
    @Test
    fun `an explicit offer registered first still leaves a later discovered one findable`() = withCleanBackends {
        registerBackend(FakeSparseLu("configured", priority = 10))
        BackendRegistry.registerAutomatic(FakeSparseLu("discovered", priority = 30))

        assertEquals("configured", koblas.generalSparseLu.name)
        assertEquals("discovered", backendNamed("discovered", Capabilities.generalSparseLu)?.name)
    }
}
