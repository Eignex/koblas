package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.assertAliasedDestinationSolves
import com.eignex.koblas.sparse.assertControlArrayKeepsUmfpackDefaults
import com.eignex.koblas.sparse.assertFactorizeGateFallsBackToReference
import com.eignex.koblas.sparse.assertGatedFallbackStillSolves
import com.eignex.koblas.sparse.assertNativeEquilibrationAndUnsupportedDropToleranceIsRejected
import com.eignex.koblas.sparse.assertReciprocalPivotConditionEstimateIsBounded
import com.eignex.koblas.sparse.assertRegistersAsTheSparseLuHalf
import com.eignex.koblas.sparse.assertRepeatedFactorizationsSurvive
import com.eignex.koblas.sparse.assertSingularIsReportedWithUnknownPosition
import com.eignex.koblas.sparse.assertSolvesAgreeWithReference
import com.eignex.koblas.sparse.sparseConformanceSystem
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import java.lang.ref.PhantomReference
import java.lang.ref.ReferenceQueue
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@Category(HostLibraryTest::class)
class UmfpackConformanceTest {

    // Every assertion below is about the native binding, and these systems are small enough that the
    // stored-entry gate would hand them to the portable factorization instead.
    private val umfpack = UmfpackSparseLu(UmfpackConfig(factorizeMin = 0))

    private fun requireSuiteSparse() {
        Assume.assumeTrue("SuiteSparse is not installed; umfpack conformance cannot run", umfpack.isAvailable)
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
    fun `the reciprocal pivot condition estimate is bounded`() {
        requireSuiteSparse()
        assertReciprocalPivotConditionEstimateIsBounded(umfpack)
    }

    @Test
    fun `a singular matrix is reported singular with an unknown position`() {
        requireSuiteSparse()
        assertSingularIsReportedWithUnknownPosition(umfpack)
    }

    @Test
    fun `equilibrate stays native and an unsupported drop tolerance is rejected`() {
        requireSuiteSparse()
        assertNativeEquilibrationAndUnsupportedDropToleranceIsRejected(umfpack) { it is UmfpackFactorization }
    }

    @Test
    fun `below its gate the portable factorization answers`() {
        requireSuiteSparse()
        assertFactorizeGateFallsBackToReference(
            { min -> UmfpackSparseLu(UmfpackConfig(factorizeMin = min)) },
            { it is UmfpackFactorization },
        )
    }

    @Test
    fun `a gated fallback still agrees with the reference`() {
        requireSuiteSparse()
        assertGatedFallbackStillSolves { min -> UmfpackSparseLu(UmfpackConfig(factorizeMin = min)) }
    }

    @Test
    fun `it registers as the sparse factorization half and reports fill`() {
        requireSuiteSparse()
        assertRegistersAsTheSparseLuHalf(umfpack, n = 20)
    }

    @Test
    fun `repeated factorizations do not exhaust native memory`() {
        requireSuiteSparse()
        assertRepeatedFactorizationsSurvive(umfpack)
    }

    @Test
    fun `the control array keeps UMFPACK's defaults with refinement off`() {
        requireSuiteSparse()
        assertControlArrayKeepsUmfpackDefaults(umfpack.refinementSteps, umfpack.pivotTolerance)
    }

    /**
     * The factorization reaches its native factors through a raw address, and the cleaner that frees them
     * holds the handles alone, so nothing the call itself holds keeps the factorization reachable. Each
     * iteration here uses one as a temporary whose last use is the receiver of the solve, which is the
     * shape a missing reachability fence would lose, and allocates between solves so collection actually
     * runs.
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
            if (round % 10 == 0) System.gc()
        }
    }

    /** A dropped factorization, watched through a queue so the caller's frame keeps no reference to it. */
    private fun watchDropped(a: F64SparseMatrix, queue: ReferenceQueue<Any>): PhantomReference<Any> =
        PhantomReference(umfpack.factor(a), queue)

    /**
     * The cleaner frees UMFPACK's factors, and only reaching them once the factorization is unreachable
     * does that. A free built from a member of the factorization captures it instead, which keeps it
     * strongly reachable from the cleaner and leaks every factorization for the life of the process.
     * Nothing observes the leak, so this watches for the collection the free waits on.
     */
    @Test
    @Suppress("ExplicitGarbageCollectionCall") // a reachability test has to provoke the collection it watches
    fun `a dropped factorization becomes unreachable so its factors can be freed`() {
        requireSuiteSparse()
        val queue = ReferenceQueue<Any>()
        val watch = watchDropped(sparseConformanceSystem(8, Random(20260824)), queue)
        // A collection is a request, not an instruction, and the reference is enqueued by another thread
        // afterwards, so this asks more than once and waits on the queue instead of polling it.
        var collected = false
        for (attempt in 0 until 5) {
            System.gc()
            if (queue.remove(ENQUEUE_WAIT_MILLIS) != null) {
                collected = true
                break
            }
        }
        watch.clear()
        assertTrue(collected, "a dropped factorization must become unreachable or its native factors never free")
    }

    private companion object {
        /** How long one attempt waits for the reference handler, five of which bound the test. */
        const val ENQUEUE_WAIT_MILLIS = 100L
    }
}
