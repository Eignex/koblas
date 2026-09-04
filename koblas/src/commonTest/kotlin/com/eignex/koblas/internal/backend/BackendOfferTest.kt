package com.eignex.koblas.internal.backend

import com.eignex.koblas.BackendRole
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.registeredBackendNames
import com.eignex.koblas.sparse.F64GeneralSparseLu
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64RepeatedSparseLu
import com.eignex.koblas.sparse.F64SparseDecompositions
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.withCleanBackends
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

/**
 * What a deployment pin means when discovery finds the backend it names. Pinning a half is a request for
 * that backend there, so it is answered by whether the backend implements the half and nothing else.
 */
class BackendOfferTest {

    /** Shaped like KLU: a host library whose ordinary LU is its repeated-pattern specialization's. */
    private class Specialized(override val name: String = "klu") :
        F64SparseDecompositions by F64ReferenceSparseLinearAlgebra,
        F64GeneralSparseLu,
        F64RepeatedSparseLu {
        override val priority: Int get() = 50
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = true

        override fun refactor(previous: F64SparseLuFactorization, a: F64SparseMatrix): F64SparseLuFactorization =
            factor(a)
    }

    private fun unpinned(): Map<BackendSlot, String?> = BackendSlot.entries.associateWith { null }

    @Test
    fun `a half pinned to a specialized provider is filled by it`() = withCleanBackends {
        val klu = Specialized()
        val pinned = unpinned() + (BackendSlot.F64GeneralSparseLu to "klu")

        registerIfOffered(klu, pinned)

        assertSame(klu, koblas.generalSparseLu, "the deployment asked for it here")
        assertEquals(listOf("klu"), registeredBackendNames(BackendRole.SPARSE_GENERAL_LU))
    }

    /** Naming a half is the only thing that sets the policy aside, so an unpinned pass still applies it. */
    @Test
    fun `an unpinned specialized provider leaves the general half alone`() = withCleanBackends {
        val klu = Specialized()

        registerIfOffered(klu, unpinned())

        assertEquals("reference", koblas.generalSparseLu.name)
        assertSame(klu, koblas.repeatedSparseLu, "its own half is still filled")
    }

    /** A pin is not a cast: a backend named for a half it does not implement still does not fill it. */
    @Test
    fun `a half pinned to a backend that does not implement it stays unfilled`() = withCleanBackends {
        val klu = Specialized()
        val pinned = unpinned() + (BackendSlot.F64SparseCholesky to "klu")

        registerIfOffered(klu, pinned)

        assertEquals("reference", koblas.sparseCholesky.name)
        assertEquals(emptyList(), registeredBackendNames(BackendRole.SPARSE_CHOLESKY))
    }

    @Test
    fun `an offer names only the halves the pin named`() {
        val offered = offerFor("klu", unpinned() + (BackendSlot.F64GeneralSparseLu to "klu"))

        assertEquals(setOf(BackendSlot.F64GeneralSparseLu), offered.named)
        assertEquals(BackendSlot.entries.toSet(), offered.halves, "the rest were left to it")
    }
}
