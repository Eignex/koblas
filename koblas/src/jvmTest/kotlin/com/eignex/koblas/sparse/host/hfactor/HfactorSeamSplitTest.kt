package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.Backend
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.sparse.GeneralSparseLu
import com.eignex.koblas.sparse.assertStrictNativeSolveAllocationContract
import com.eignex.koblas.sparse.basis.BasisSolvers
import kotlin.test.*

/**
 * The basis solver seam is HFactor's, and the general LU role is one it fills below every other host. These
 * pin that arrangement: nothing here calls a library, so they hold wherever the tests run.
 */
class HfactorSeamSplitTest {
    private val hfactor = HfactorSparseLu()

    // Read as a backend, so the half it offers is a question about the object rather than one the
    // compiler has already answered from the declared type.
    private val hfactorBackend: Backend = hfactor

    @Test
    fun `the HFactor backend offers basis solvers`() {
        assertTrue(hfactorBackend is BasisSolvers, "HFactor is what the basis solver seam is for")
    }

    @Test
    fun `HFactor offers the general sparse LU role`() {
        assertTrue(hfactorBackend is GeneralSparseLu)
    }

    @Test
    fun `HFactor ranks below every host sparse LU`() {
        assertTrue(hfactor.priority < HOST_BACKEND_PRIORITY, "HFactor priority ${hfactor.priority}")
    }

    @Test
    fun `repeated solves declare a strict allocation contract where HFactor is installed`() {
        if (!hfactor.isAvailable) return
        assertStrictNativeSolveAllocationContract(hfactor)
    }
}
