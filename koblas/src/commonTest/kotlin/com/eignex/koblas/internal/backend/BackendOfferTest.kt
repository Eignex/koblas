package com.eignex.koblas.internal.backend

import com.eignex.koblas.BackendRole
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.registeredBackendNames
import com.eignex.koblas.sparse.GeneralSparseLu
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.SparseLuFactorization
import com.eignex.koblas.withCleanBackends
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

/**
 * What a deployment pin means when discovery finds the backend it names. Pinning a half is a request for
 * that backend there, so it is answered by whether the backend implements the half and nothing else.
 */
class BackendOfferTest {

    /** A host library whose ordinary LU is its repeated-pattern specialization's. */
    private class Specialized(override val name: String = "specialized") : GeneralSparseLu {
        override val priority: Int get() = 50
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = true

        override fun factor(a: SparseMatrix): SparseLuFactorization = ReferenceSparseLinearAlgebra.factor(a)
    }

    private fun unpinned(): Map<BackendSlot, String?> = BackendSlot.entries.associateWith { null }

    @Test
    fun `a half pinned to a specialized provider is filled by it`() = withCleanBackends {
        val specialized = Specialized()
        val pinned = unpinned() + (BackendSlot.GeneralSparseLu to "specialized")

        registerIfOffered(specialized, pinned)

        assertSame(specialized, koblas.generalSparseLu, "the deployment asked for it here")
        assertEquals(listOf("specialized"), registeredBackendNames(BackendRole.SPARSE_GENERAL_LU))
    }

    @Test
    fun `an offer names only the halves the pin named`() {
        val offered = offerFor(
            namedProvider("specialized"),
            unpinned() + (BackendSlot.GeneralSparseLu to "specialized"),
        )

        assertEquals(setOf(BackendSlot.GeneralSparseLu), offered.named)
        assertEquals(BackendSlot.entries.toSet(), offered.halves, "the rest were left to it")
    }
}
