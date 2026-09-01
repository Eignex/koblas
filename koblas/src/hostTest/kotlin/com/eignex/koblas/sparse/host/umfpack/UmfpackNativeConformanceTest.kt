// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.discoverBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.cholmod.CholmodCholeskyFactorization
import com.eignex.koblas.sparse.host.cholmod.CholmodLdlFactorization
import kotlin.random.Random
import kotlin.test.*

/** Checks the native UMFPACK bindings against the reference implementation. */
class UmfpackNativeConformanceTest {

    /** Required rather than skipped, since Kotlin/Native has no Assume and a skipped suite reads as green. */
    // Ungated for the reason the JVM conformance test is: these assertions are about the native binding.
    private val umfpack: UmfpackSparseLu = UmfpackSparseLu(UmfpackConfig()).also {
        require(it.isAvailable) { "host SuiteSparse expected in the test environment" }
    }

    @Test
    fun `the binding resolves and registers`() {
        discoverBackends()
        assertEquals("umfpack", umfpack.name)
        assertEquals(HOST_BACKEND_PRIORITY, umfpack.priority, "every koblas host binding registers at one priority")
        val discovered = F64SparseBackends()
        assertEquals("umfpack", koblas.generalSparseLu.name, "UMFPACK should remain the general LU default")
        assertEquals(
            discovered.klu.takeIf { it.isAvailable }?.name,
            koblas.repeatedSparseLu?.name,
            "KLU should fill only the repeated-pattern LU role",
        )
        assertEquals(
            discovered.basiclu.takeIf { it.isAvailable }?.name ?: "reference",
            koblas.basisFactorizations.name,
            "BASICLU should fill only the basis-factorization role",
        )
        val symmetric = listOf(discovered.umfpack, discovered.klu)
            .filter { it.isAvailable }
            .maxBy { it.priority }
        assertEquals(symmetric.name, koblas.sparseCholesky.name)
        assertEquals(symmetric.name, koblas.quasiDefiniteLdl.name)
    }

    @Test
    fun `solutions agree with the portable factorization in both directions`() = assertSolvesAgreeWithReference(umfpack)

    @Test
    fun `an aliased destination still solves correctly`() = assertAliasedDestinationSolves(umfpack)

    @Test
    fun `repeated solves declare a strict allocation contract`() = assertStrictNativeSolveAllocationContract(umfpack)

    @Test
    fun `a native LU factor closes deterministically`() {
        val factorization = umfpack.factor(sparseConformanceSystem(8, Random(20261007)))
        assertIs<UmfpackFactorization>(factorization)

        assertNativeFactorCloseContract(factorization)
    }

    @Test
    fun `a native CHOLMOD factor closes deterministically`() {
        val factorization = umfpack.cholesky(sparseSymmetricConformanceSystem(12, Random(20261008)))
        assertIs<CholmodCholeskyFactorization>(factorization)

        assertNativeFactorCloseContract(factorization)
    }

    @Test
    fun `native CHOLMOD solves reuse their descriptor scratch`() {
        val n = 18
        val factor = umfpack.cholesky(sparseSymmetricConformanceSystem(n, Random(20261010)))
        assertIs<CholmodCholeskyFactorization>(factor)

        assertStrictNativeSolveAllocationContract(factor, DoubleArray(n) { it * 0.25 - 1.0 })
    }

    /** CHOLMOD ships beside UMFPACK, so the collection answers this seam's Cholesky natively here too. */
    @Test
    fun `the Cholesky comes from CHOLMOD beside it`() {
        assertIs<CholmodCholeskyFactorization>(
            umfpack.cholesky(sparseSymmetricConformanceSystem(24, Random(20260916))),
            "expected CHOLMOD's factorization rather than the portable fallback",
        )
        assertCholeskyAgreesWithReference(umfpack)
    }

    @Test
    fun `the QR reaches SPQR or the portable factorization and agrees either way`() {
        assertQrAgreesWithReference(umfpack)
    }

    @Test
    fun `multiple right hand sides use the CHOLMOD block solve`() {
        val a = sparseSymmetricConformanceSystem(18, Random(20260831))

        assertNativeBlockFactorSolvesAgreeWithReference(
            umfpack.cholesky(a),
            F64ReferenceSparseLinearAlgebra.cholesky(a),
        )
    }

    /**
     * The factorization CHOLMOD produces by default, which the Cholesky path turns off. An indefinite matrix
     * is what tells the two apart: it has an `L·D·Lᵀ` and no `L·Lᵀ`.
     */
    @Test
    fun `the LDL comes from CHOLMOD and factors a matrix the Cholesky refuses`() {
        val a = indefiniteConformanceSystem(20, Random(20260929))

        val f = umfpack.quasiDefiniteLdl(a)

        assertIs<CholmodLdlFactorization>(f, "expected CHOLMOD's factorization rather than the portable fallback")
        assertTrue(!f.singular, "an invertible indefinite system has an L D Lt")
        assertFailsWith<NotPositiveDefinite> { umfpack.cholesky(a) }
    }

    @Test
    fun `LDL solutions agree with the portable one`() = assertLdlAgreesWithReference(umfpack)

    @Test
    fun `the reciprocal pivot condition estimate is bounded`() =
        assertReciprocalPivotConditionEstimateIsBounded(umfpack)

    @Test
    fun `a singular matrix is reported singular with an unknown position`() =
        assertSingularIsReportedWithUnknownPosition(umfpack)

    @Test
    fun `a backend set to equilibrate stays native and still solves`() =
        assertNativeEquilibration(UmfpackSparseLu(UmfpackConfig(equilibrate = true))) {
            it is UmfpackFactorization
        }

    @Test
    fun `it registers as the sparse factorization half and reports fill`() =
        assertRegistersAsTheSparseLuHalf(umfpack, n = 20)

    @Test
    fun `the control array keeps UMFPACK's defaults with refinement off`() =
        assertControlArrayKeepsUmfpackDefaults(umfpack.refinementSteps, umfpack.pivotTolerance)

    @Test
    fun `shared options reach the effective UMFPACK control array`() {
        val configured = UmfpackSparseLu(
            UmfpackConfig(
                libraryPath = null,
                options = UmfpackOptions(
                    equilibrate = true,
                    iterativeRefinementSteps = 3,
                    pivotTolerance = 0.2,
                    scaling = UmfpackScaling.MAX,
                ),
            ),
        )

        assertEquals(3.0, configured.refinementSteps)
        assertEquals(0.2, configured.pivotTolerance)
        assertEquals(2.0, configured.scaling)
    }

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
