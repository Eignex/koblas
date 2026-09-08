package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * The context the seams resolve to is assembled on demand and cached against a change count. A registration
 * into one half therefore has to invalidate a context cached from an earlier read, and must not cost the
 * registrations already in the other halves.
 */
class ResolvedContextTest {

    private class BlasHalf(override val name: String) : F64Blas by F64ReferenceBlas {
        override val priority: Int get() = 40
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = true
    }

    @Test
    fun `clearing the registry is visible after the context has already been read`() = withCleanBackends {
        registerBackend(BlasHalf("someblas"))
        assertEquals("someblas", koblas.blas.name)
        resetBackends()
        assertEquals("reference", koblas.blas.name, "the cached context outlived the registration")
    }

    @Test
    fun `both assembly paths fill every backend role`() {
        // The registry and the builder each enumerate the backend halves and sparse roles
        // independently, which BackendSlot's own KDoc warns about: a half described in two places is a half
        // the two paths can answer differently. Adding one and editing only the other compiles fine, so this
        // is what fails instead.
        val fromRegistry = koblas
        val fromBuilder = F64ContextBuilder().resolve()

        for (role in BackendRole.entries) {
            assertNotNull(fromRegistry.backendFor(role), "the registry left $role unfilled")
            assertNotNull(fromBuilder.backendFor(role), "the builder left $role unfilled")
        }
        assertEquals(BackendRole.entries.size, fromRegistry.status.backends.size)
        assertEquals(BackendRole.entries.size, fromBuilder.status.backends.size)

        // The sparse roles are the ones held separately from the composed surface, so they are the ones a
        // path can silently drop.
        for (context in listOf(fromRegistry, fromBuilder)) {
            assertNotNull(context.generalSparseLu)
            assertNotNull(context.sparseCholesky)
            assertNotNull(context.quasiDefiniteLdl)
            assertNotNull(context.sparseQr)
            assertNotNull(context.basisFactorizations)
        }
    }
}
