package com.eignex.koblas

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.*
import kotlin.test.*

class SparseSemanticRolesTest {

    private open class LegacyProvider(override val name: String, override val priority: Int) :
        SparseLapack by ReferenceSparseLinearAlgebra {
        override val isPortable: Boolean get() = false
    }

    private class General(name: String = "general", priority: Int = 10) :
        LegacyProvider(name, priority),
        GeneralSparseLu

    private class Repeated(name: String = "repeated", priority: Int = 100) :
        LegacyProvider(name, priority),
        GeneralSparseLu,
        RepeatedSparseLu {
        override fun refactor(previous: SparseLuFactorization, a: SparseMatrix): SparseLuFactorization = factor(a)
    }

    private class Basis(name: String = "basis", priority: Int = 200) :
        LegacyProvider(name, priority),
        GeneralSparseLu,
        BasisFactorizations {
        override fun factorBasis(basis: SparseMatrix): BasisFactorization =
            ReferenceSparseLinearAlgebra.factorBasis(basis)
    }

    private class Complete :
        LegacyProvider("complete", priority = 10),
        GeneralSparseLu,
        SparseCholesky,
        QuasiDefiniteLdl,
        SparseQr

    private class CholeskyOnly :
        LegacyProvider("cholesky-only", priority = 10),
        SparseCholesky

    private class LdlOnly :
        LegacyProvider("ldl-only", priority = 10),
        QuasiDefiniteLdl

    private class QrOnly :
        LegacyProvider("qr-only", priority = 10),
        SparseQr

    /**
     * Shaped like koblas's own reference: portable, and filling a specialized half beside a general LU that
     * really is general. What makes a provider specialized is calling out to a library built for one job.
     */
    private class PortableComplete :
        LegacyProvider("portable-complete", priority = 10),
        GeneralSparseLu,
        BasisFactorizations {
        override val isPortable: Boolean get() = true

        override fun factorBasis(basis: SparseMatrix): BasisFactorization =
            ReferenceSparseLinearAlgebra.factorBasis(basis)
    }

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
        assertSame(repeated, koblas.capability(Capabilities.repeatedSparseLu))
        assertSame(basis, koblas.capability(Capabilities.basisFactorizations))
        assertSame(repeated, backendNamed("repeated", Capabilities.repeatedSparseLu))
        assertSame(basis, backendNamed("basis", Capabilities.basisFactorizations))
        assertNull(backendNamed("repeated", Capabilities.generalSparseLu))
        assertNull(backendNamed("basis", Capabilities.generalSparseLu))
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
        assertEquals("reference", koblas.quasiDefiniteLdl.name)
        assertEquals("reference", koblas.sparseQr.name)
    }

    @Test
    fun `a provider that fills no role is rejected by an explicit context`() {
        val provider = LegacyProvider("third-party", priority = 50)

        assertFailsWith<IllegalArgumentException> {
            ContextBuilder().withBackend(BackendRole.SPARSE_GENERAL_LU, provider)
        }
    }

    @Test
    fun `a complete backend selects all four factorization providers`() {
        val provider = Complete()

        val context = ContextBuilder()
            .withBackend(provider)
            .resolve()

        assertSame(provider, context.generalSparseLu)
        assertSame(provider, context.sparseCholesky)
        assertSame(provider, context.quasiDefiniteLdl)
        assertSame(provider, context.sparseQr)
        assertNotSame(provider, context.sparseDecompositions)
    }

    @Test
    fun `a configured portable decomposition fills every factorization role`() {
        val provider = ReferenceSparseDecompositions(equilibrate = true)

        val context = ContextBuilder()
            .withBackend(provider)
            .resolve()

        assertSame(provider, context.generalSparseLu)
        assertSame(provider, context.sparseCholesky)
        assertSame(provider, context.quasiDefiniteLdl)
        assertSame(provider, context.sparseQr)
    }

    @Test
    fun `explicit contexts resolve semantic providers independently`() {
        val general = General()
        val repeated = Repeated()
        val basis = Basis()

        val context = ContextBuilder()
            .withBackend(BackendRole.SPARSE_GENERAL_LU, general)
            .withBackend(BackendRole.SPARSE_REPEATED_LU, repeated)
            .withBackend(BackendRole.BASIS_FACTORIZATIONS, basis)
            .resolve()

        assertSame(general, context.generalSparseLu)
        assertSame(repeated, context.repeatedSparseLu)
        assertSame(basis, context.basisFactorizations)
        assertEquals("general", context.sparseDecompositions.generalLuProviderName())
    }

    /** The same choice registration makes, so a context resolved either way fills the role the same way. */
    @Test
    fun `specialized providers do not change general sparse LU in an explicit context`() {
        val repeated = Repeated()
        val basis = Basis()

        val context = ContextBuilder()
            .withBackend(repeated)
            .withBackend(basis)
            .resolve()

        assertSame(repeated, context.repeatedSparseLu)
        assertSame(basis, context.basisFactorizations)
        assertEquals("reference", context.generalSparseLu.name)
    }

    /** Each half comes out of its own seam, which is what a resolved context wiring two of them alike loses. */
    @Test
    fun `each factorization role resolves the provider registered for it`() = withCleanBackends {
        val cholesky = CholeskyOnly()
        val ldl = LdlOnly()
        val qr = QrOnly()

        registerBackend(cholesky)
        registerBackend(ldl)
        registerBackend(qr)

        assertSame(cholesky, koblas.sparseCholesky)
        assertSame(ldl, koblas.quasiDefiniteLdl)
        assertSame(qr, koblas.sparseQr)
        assertEquals("reference", koblas.generalSparseLu.name, "nothing filled the general half")
    }

    @Test
    fun `a portable provider filling a specialized half keeps general sparse LU`() = withCleanBackends {
        val provider = PortableComplete()

        registerBackend(provider)

        assertSame(provider, koblas.generalSparseLu)
        assertSame(provider, koblas.basisFactorizations)
    }

    @Test
    fun `a portable provider filling a specialized half keeps general LU in an explicit context`() {
        val provider = PortableComplete()

        val context = ContextBuilder()
            .withBackend(provider)
            .resolve()

        assertSame(provider, context.generalSparseLu)
        assertSame(provider, context.basisFactorizations)
    }

    @Test
    fun `a specialized provider can be explicitly selected for general LU`() {
        val repeated = Repeated()

        val context = ContextBuilder()
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

private fun SparseLapack.generalLuProviderName(): String = (this as SparseDecompositionRoles).generalLu.name
