// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.host.F64SparseBackends
import kotlin.random.Random
import kotlin.test.*

/** Checks the native UMFPACK bindings against the reference implementation. */
class UmfpackNativeConformanceTest {

    /** Required rather than skipped, since Kotlin/Native has no Assume and a skipped suite reads as green. */
    // Ungated for the reason the JVM conformance test is: these assertions are about the native binding.
    private val umfpack: UmfpackSparseLu = UmfpackSparseLu(UmfpackConfig(factorizeMin = 0)).also {
        require(it.isAvailable) { "host SuiteSparse expected in the test environment" }
    }

    @Test
    fun `the binding resolves and registers`() {
        assertEquals("umfpack", umfpack.name)
        assertEquals(HOST_BACKEND_PRIORITY, umfpack.priority, "every koblas host binding registers at one priority")
        // KLU and BASICLU register here too where a host has them, and rank above UMFPACK, so what
        // discovery is asserted to have found is the strongest binding this machine offers.
        val strongest = F64SparseBackends().let { listOf(it.umfpack, it.klu, it.basiclu) }
            .filter { it.isAvailable }
            .maxBy { it.priority }
        assertEquals(strongest.name, koblas.sparseLu.name, "discovery should have registered the backend")
    }

    @Test
    fun `solutions agree with the portable factorization in both directions`() = assertSolvesAgreeWithReference(umfpack)

    @Test
    fun `an aliased destination still solves correctly`() = assertAliasedDestinationSolves(umfpack)

    @Test
    fun `the reciprocal pivot condition estimate is bounded`() =
        assertReciprocalPivotConditionEstimateIsBounded(umfpack)

    @Test
    fun `a singular matrix is reported singular with an unknown position`() =
        assertSingularIsReportedWithUnknownPosition(umfpack)

    @Test
    fun `a backend set to equilibrate stays native and still solves`() =
        assertNativeEquilibration(UmfpackSparseLu(UmfpackConfig(factorizeMin = 0, equilibrate = true))) {
            it is UmfpackFactorization
        }

    @Test
    fun `it registers as the sparse factorization half and reports fill`() =
        assertRegistersAsTheSparseLuHalf(umfpack, n = 20)

    @Test
    fun `the control array keeps UMFPACK's defaults with refinement off`() =
        assertControlArrayKeepsUmfpackDefaults(umfpack.refinementSteps, umfpack.pivotTolerance)

    @Test
    fun `repeated factorizations do not exhaust native memory`() = assertRepeatedFactorizationsSurvive(umfpack)

    /**
     * The anchor holds a factorization reachable for the length of a native call and has to let go
     * afterwards. Left set, it would pin every factorization the thread ever solved against, turning a rare
     * crash into a steady leak.
     */
    @Test
    fun `the call anchor is released once the call returns`() {
        val a = sparseConformanceSystem(12, Random(20260922))
        val f = umfpack.factor(a)
        assertNull(AnchoredFactorization.held, "nothing should be anchored before a call")
        f.solve(DoubleArray(12) { 1.0 })
        assertNull(AnchoredFactorization.held, "solve left its factorization anchored")
    }
}
