package com.eignex.koblas

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.*
import kotlin.test.*

class SparseSemanticRolesTest {

    private open class LegacyProvider(override val name: String, override val priority: Int) :
        F64SparseDecompositions by F64ReferenceSparseLinearAlgebra {
        override val isPortable: Boolean get() = false
    }

    private class General(name: String = "general", priority: Int = 10) :
        LegacyProvider(name, priority),
        F64GeneralSparseLu

    private class Repeated(name: String = "repeated", priority: Int = 100) :
        LegacyProvider(name, priority),
        F64GeneralSparseLu,
        F64RepeatedSparseLu {
        override fun refactor(previous: F64SparseLuFactorization, a: F64SparseMatrix): F64SparseLuFactorization =
            factor(a)
    }

    private class Basis(name: String = "basis", priority: Int = 200) :
        LegacyProvider(name, priority),
        F64GeneralSparseLu,
        F64BasisFactorizations {
        override fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization =
            F64ReferenceSparseLinearAlgebra.factorBasis(basis)
    }

    private class Complete :
        LegacyProvider("complete", priority = 10),
        F64GeneralSparseLu,
        F64SparseCholesky,
        F64SparseLdl,
        F64SparseQr

    @Test
    fun `specialized providers do not change general sparse LU`() = withCleanBackends {
        val general = General()
        val repeated = Repeated()
        val basis = Basis()

        registerBackend(general)
        registerBackend(repeated)
        registerBackend(basis)

        assertSame(general, koblas.generalSparseLu)
        assertSame(repeated, koblas.repeatedSparseLu)
        assertSame(basis, koblas.basisFactorizations)
        assertSame(repeated, koblas.capability(F64Capabilities.repeatedSparseLu))
        assertSame(basis, koblas.capability(F64Capabilities.basisFactorizations))
        assertSame(repeated, backendNamed("repeated", F64Capabilities.repeatedSparseLu))
        assertSame(basis, backendNamed("basis", F64Capabilities.basisFactorizations))
        assertNull(backendNamed("repeated", F64Capabilities.generalSparseLu))
        assertNull(backendNamed("basis", F64Capabilities.generalSparseLu))
        assertEquals(listOf("general"), registeredBackendNames(BackendRole.SPARSE_GENERAL_LU))
        assertEquals(listOf("repeated"), registeredBackendNames(BackendRole.SPARSE_REPEATED_LU))
        assertEquals(listOf("basis"), registeredBackendNames(BackendRole.BASIS_FACTORIZATIONS))
    }

    /**
     * A backend offers the roles it implements. Satisfying the wide seam alone offers nothing, so a provider
     * that names no role leaves every one of them to the reference rather than silently taking all of them.
     */
    @Test
    fun `a provider that fills no role is rejected by registration`() = withCleanBackends {
        assertFailsWith<IllegalArgumentException> {
            registerBackend(LegacyProvider("third-party", priority = 50))
        }

        assertEquals("reference", koblas.generalSparseLu.name)
        assertEquals("reference", koblas.sparseCholesky.name)
        assertEquals("reference", koblas.sparseLdl.name)
        assertEquals("reference", koblas.sparseQr.name)
    }

    @Test
    fun `a provider that fills no role is rejected by an explicit context`() {
        val provider = LegacyProvider("third-party", priority = 50)

        assertFailsWith<IllegalArgumentException> {
            F64ContextBuilder().withBackend(BackendRole.SPARSE_GENERAL_LU, provider)
        }
    }

    @Test
    fun `a complete backend selects all four factorization providers`() {
        val provider = Complete()

        val context = F64ContextBuilder()
            .withBackend(provider)
            .resolve()

        assertSame(provider, context.generalSparseLu)
        assertSame(provider, context.sparseCholesky)
        assertSame(provider, context.sparseLdl)
        assertSame(provider, context.sparseQr)
        assertNotSame(provider, context.sparseDecompositions)
    }

    @Test
    fun `a configured portable decomposition fills every factorization role`() {
        val provider = F64ReferenceSparseDecompositions(equilibrate = true)

        val context = F64ContextBuilder()
            .withBackend(provider)
            .resolve()

        assertSame(provider, context.generalSparseLu)
        assertSame(provider, context.sparseCholesky)
        assertSame(provider, context.sparseLdl)
        assertSame(provider, context.sparseQr)
    }

    @Test
    fun `explicit contexts resolve semantic providers independently`() {
        val general = General()
        val repeated = Repeated()
        val basis = Basis()

        val context = F64ContextBuilder()
            .withBackend(BackendRole.SPARSE_GENERAL_LU, general)
            .withBackend(BackendRole.SPARSE_REPEATED_LU, repeated)
            .withBackend(BackendRole.BASIS_FACTORIZATIONS, basis)
            .resolve()

        assertSame(general, context.generalSparseLu)
        assertSame(repeated, context.repeatedSparseLu)
        assertSame(basis, context.basisFactorizations)
        assertEquals("general", context.sparseDecompositions.generalLuProviderName())
    }

    @Test
    fun `a specialized provider can be explicitly selected for general LU`() {
        val repeated = Repeated()

        val context = F64ContextBuilder()
            .withBackend(BackendRole.SPARSE_GENERAL_LU, repeated)
            .resolve()

        assertSame(repeated, context.generalSparseLu)
    }

    @Test
    fun `an empty registry reports repeated pattern LU as unavailable`() = withCleanBackends {
        assertNull(koblas.repeatedSparseLu)
        val status = koblas.status[BackendRole.SPARSE_REPEATED_LU]
        assertFalse(status.available)
        assertEquals("unavailable", status.provider)
    }
}

private fun F64SparseDecompositions.generalLuProviderName(): String =
    (this as F64SparseDecompositionRoles).generalLu.name
