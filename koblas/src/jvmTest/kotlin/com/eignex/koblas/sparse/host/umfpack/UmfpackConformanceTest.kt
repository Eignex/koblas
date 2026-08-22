package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.assertClose
import com.eignex.koblas.sparse.assertAliasedDestinationSolves
import com.eignex.koblas.sparse.assertControlArrayKeepsUmfpackDefaults
import com.eignex.koblas.sparse.assertDeterminantAgreesWithReference
import com.eignex.koblas.sparse.assertEmptyAndZeroMatricesTakeThePortablePath
import com.eignex.koblas.sparse.assertRegistersAsTheSparseLapackHalf
import com.eignex.koblas.sparse.assertRepeatedFactorizationsSurvive
import com.eignex.koblas.sparse.assertSingularIsReportedWithUnknownPosition
import com.eignex.koblas.sparse.assertSolvesAgreeWithReference
import com.eignex.koblas.sparse.assertUnsupportedRequestsFallBack
import com.eignex.koblas.sparse.sparseConformanceSystem
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
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
    fun `the control array keeps UMFPACK's defaults with refinement off`() {
        requireSuiteSparse()
        assertControlArrayKeepsUmfpackDefaults(UmfpackCalls.refinementSteps, UmfpackCalls.pivotTolerance)
    }

    /**
     * The factorization reaches its native factors through a raw address, and the cleaner that frees them
     * captures the handles rather than the factorization, so nothing the call itself holds keeps the
     * factorization reachable. Each iteration here uses one as a temporary whose last use is the receiver
     * of the solve, which is the shape a missing reachability fence would lose, and allocates between
     * solves so collection actually runs.
     */
    @Test
    fun `a factorization used as a temporary survives collection during its own solve`() {
        requireSuiteSparse()
        val n = 24
        val a = sparseConformanceSystem(n, Random(20260821))
        val b = DoubleArray(n) { 1.0 }
        val expected = umfpack.factor(a).solve(b)
        repeat(30) { round ->
            assertClose(expected, umfpack.factor(a).solve(b), "solve under collection pressure round $round")
            assertClose(
                expected,
                umfpack.factor(a).solveInto(b, DoubleArray(n)),
                "solveInto under collection pressure round $round",
            )
            umfpack.factor(a).determinant()
            if (round % 10 == 0) System.gc()
        }
    }
}
