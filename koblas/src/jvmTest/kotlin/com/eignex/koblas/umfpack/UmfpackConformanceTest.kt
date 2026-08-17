package com.eignex.koblas.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.HostLibraryTest
import com.eignex.koblas.sparse.assertAliasedDestinationSolves
import com.eignex.koblas.sparse.assertDeterminantAgreesWithReference
import com.eignex.koblas.sparse.assertEmptyAndZeroMatricesTakeThePortablePath
import com.eignex.koblas.sparse.assertRegistersAsTheSparseLapackHalf
import com.eignex.koblas.sparse.assertRepeatedFactorizationsSurvive
import com.eignex.koblas.sparse.assertSingularIsReportedWithUnknownPosition
import com.eignex.koblas.sparse.assertSolvesAgreeWithReference
import com.eignex.koblas.sparse.assertUnsupportedRequestsFallBack
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.test.Test
import kotlin.test.assertEquals

@Category(HostLibraryTest::class)
class UmfpackConformanceTest {

    private val umfpack = UmfpackSparseLapack()

    private fun requireSuiteSparse() {
        Assume.assumeTrue("SuiteSparse is not installed; umfpack conformance cannot run", UmfpackCalls.available)
    }

    @Test
    fun `the binding resolves on a machine with suitesparse`() {
        requireSuiteSparse()
        assertEquals("umfpack", umfpack.name)
        assertEquals(HOST_BACKEND_PRIORITY, umfpack.priority, "every koblas host binding registers at one priority")
    }

    @Test
    fun `solutions agree with the portable factorization in both directions`() {
        requireSuiteSparse()
        assertSolvesAgreeWithReference(umfpack)
    }

    @Test
    fun `an aliased destination still solves correctly`() {
        requireSuiteSparse()
        assertAliasedDestinationSolves(umfpack)
    }

    @Test
    fun `the determinant agrees with the portable factorization`() {
        requireSuiteSparse()
        assertDeterminantAgreesWithReference(umfpack)
    }

    @Test
    fun `a singular matrix is reported singular with an unknown position`() {
        requireSuiteSparse()
        assertSingularIsReportedWithUnknownPosition(umfpack)
    }

    @Test
    fun `equilibrate and a drop tolerance fall back to the portable factorization`() {
        requireSuiteSparse()
        assertUnsupportedRequestsFallBack(umfpack) { it is UmfpackFactorization }
    }

    @Test
    fun `it registers as the sparse factorization half and reports fill`() {
        requireSuiteSparse()
        assertRegistersAsTheSparseLapackHalf(umfpack, n = 20)
    }

    @Test
    fun `empty and all-zero matrices take the portable path`() {
        requireSuiteSparse()
        assertEmptyAndZeroMatricesTakeThePortablePath(umfpack)
    }

    @Test
    fun `repeated factorizations do not exhaust native memory`() {
        requireSuiteSparse()
        assertRepeatedFactorizationsSurvive(umfpack)
    }

    @Test
    fun `solves run without iterative refinement`() {
        requireSuiteSparse()
        assertEquals(0.0, UmfpackCalls.refinementSteps, "umfpack solves must not refine")
    }

    @Test
    fun `the rest of the control array keeps UMFPACK's defaults`() {
        requireSuiteSparse()
        assertEquals(0.1, UmfpackCalls.pivotTolerance, "umfpack_di_defaults did not fill the control array")
    }
}
