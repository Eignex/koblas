package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.BackendRole
import com.eignex.koblas.F64Capabilities
import com.eignex.koblas.F64ContextBuilder
import com.eignex.koblas.backendNamed
import com.eignex.koblas.koblas
import com.eignex.koblas.registerBackend
import com.eignex.koblas.registeredBackendNames
import com.eignex.koblas.testutil.host.HostLibraryTest
import com.eignex.koblas.withCleanBackends
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.test.*

/**
 * The specialization policy against the library it was written for. KLU refactors one sparsity pattern, so
 * the ordinary LU it offers is that specialization's factorization rather than a general one, and an offer of
 * everything it implements leaves the general half alone. Elsewhere this shape is covered by fakes.
 */
@Category(HostLibraryTest::class)
class KluRoleSelectionTest {
    private val klu = KluSparseLu(KluConfig())

    @Test
    fun `registration leaves general sparse LU to the reference`() {
        Assume.assumeTrue("KLU is not installed; selection cannot run", klu.isAvailable)

        withCleanBackends {
            registerBackend(klu)

            assertEquals("klu", koblas.repeatedSparseLu?.name, "the half its specialization is about")
            assertEquals("reference", koblas.generalSparseLu.name)
            assertEquals(emptyList(), registeredBackendNames(BackendRole.SPARSE_GENERAL_LU))
            assertNull(backendNamed("klu", F64Capabilities.generalSparseLu))
        }
    }

    /** The escape hatch the policy is documented against, so a caller who wants KLU here can say so. */
    @Test
    fun `an explicit context can still name KLU for general sparse LU`() {
        Assume.assumeTrue("KLU is not installed; selection cannot run", klu.isAvailable)

        val context = F64ContextBuilder()
            .withBackend(BackendRole.SPARSE_GENERAL_LU, klu)
            .resolve()

        assertSame(klu, context.generalSparseLu)
    }
}
