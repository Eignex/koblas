package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.Backend
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.sparse.assertStrictNativeSolveAllocationContract
import com.eignex.koblas.sparse.basis.F64BasisSolvers
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu
import kotlin.test.*

/**
 * The two basis contracts want different backends, and the seam split is what lets each have the one it
 * wants. These pin that arrangement: nothing here calls a library, so they hold wherever the tests run.
 */
class HfactorSeamSplitTest {
    private val hfactor = HfactorSparseLu()
    private val basiclu = BasicluSparseLu()

    // Read as backends, so the half each offers is a question about the object rather than one the
    // compiler has already answered from the declared type.
    private val hfactorBackend: Backend = hfactor
    private val basicluBackend: Backend = basiclu

    @Test
    fun `only the HFactor backend offers basis solvers`() {
        assertTrue(hfactorBackend is F64BasisSolvers, "HFactor is what the basis solver seam is for")
        assertFalse(
            basicluBackend is F64BasisSolvers,
            "BASICLU keeps the contract that takes any entering column",
        )
    }

    @Test
    fun `HFactor ranks below BASICLU for the sparse LU half`() {
        assertTrue(
            hfactor.priority < basiclu.priority,
            "HFactor ${hfactor.priority} must not take the LU half from BASICLU ${basiclu.priority}",
        )
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

    /** The column-replacement contract is BASICLU's own routine now, not a seam method HFactor declines. */
    @Test
    fun `BASICLU still claims in-place column replacement where it is available`() {
        assertEquals(basiclu.isAvailable, basiclu.supportsBasisUpdates)
    }
}
