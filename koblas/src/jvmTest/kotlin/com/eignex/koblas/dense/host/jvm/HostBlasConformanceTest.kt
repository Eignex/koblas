package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.host.*
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.cblas.OPENBLAS_SONAMES
import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.*

@Category(HostLibraryTest::class)
class HostBlasConformanceTest {

    @Test
    fun `the host BLAS consumes strided views in place`() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
        assertStridedProductsAgreeWithReference(
            F64Cblas(HostBlasConfig()),
        )
    }

    /**
     * Discovery is what registers a host binding, and reading the context does not start it, so the test
     * that asserts the outcome is the one that has to ask for it.
     */
    @Test
    fun `the host backend resolves when the machine has OpenBLAS`() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
        discoverBackends()
        assertTrue(koblasInfo.contains("openblas"), koblasInfo)
    }

    /**
     * The hosts this suite runs on ship the LP64 build, so a rejection here is the width probe misfiring
     * rather than a host it was right to turn away.
     */
    @Test
    fun `an LP64 host is not turned away by the pivot-width probe`() {
        val library = FfmLibrary.open(OPENBLAS_SONAMES, "cblas_dgemm", "the host OpenBLAS")
        Assume.assumeTrue("host CBLAS is not installed", library.present)
        assertTrue(HostBlasCalls(HostBlasConfig()).available, "the host OpenBLAS was judged ILP64")
    }

    /** A factorization at a size that engages OpenBLAS's parallel path. */
    @Test
    fun `a large factorization does not take the process down`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        val n = 512
        val rng = Random(20260807)
        val a = wellConditioned(n, rng)
        val host = F64Lapacke()
        repeat(4) {
            val lu = host.factor(a)
            assertEquals(n, lu.n)
        }
    }

    /** The level-2 half runs portable here whatever the size; the no-SIMD host pass is what reaches it. */
    @Test
    fun `the gated level 2 and 3 routines match reference`() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
        assertTriangularAgreesWithReference(F64Cblas(), intArrayOf(1, 5, 12, 24))
        assertGerAgreesWithReference(F64Cblas())
        assertSyrAgreesWithReference(F64Cblas())
    }

    /** Lowering the gates is the only way the native solve runs; the shipped values keep it portable. */
    @Test
    fun `the host solves match reference once their gates admit them`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertLuAgreesWithReference(F64Lapacke(HostBlasConfig()), intArrayOf(33, 256))
    }

    /** Both inverses read the level-3 gate, so lowering that is what reaches them at every size here. */
    @Test
    fun `the host inverses match reference once their gate admits them`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertInversesAgreeWithReference(F64Lapacke(HostBlasConfig()), intArrayOf(7, 33, 128))
    }

    @Test
    fun `symv refuses a non-square matrix`() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
        assertSymvRefusesNonSquare(F64Cblas())
    }

    @Test
    fun `a singular LDL is refused at every width`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertSingularLdlIsRefused(F64Lapacke())
    }

    @Test
    fun `level 3 matches reference at blocked sizes`() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
        assertLevel3AgreesWithReference(F64Cblas(), intArrayOf(7, 64, 256))
    }

    /** Above cholesky's gate of 32 and invertSpd's of 16, so the host path is the one under test. */
    @Test
    fun `the SPD suite matches reference where the gates open`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertSpdSuiteAgreesWithReference(F64Lapacke(), intArrayOf(33, 256))
    }

    @Test
    fun `a non positive definite input falls back to the portable path`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertNonPositiveDefiniteFallsBack(F64Lapacke(), n = 256)
    }

    @Test
    fun `pivoted QR matches reference in rank and reconstruction`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertPivotedQrAgreesWithReference(F64Lapacke(), m = 96, cols = 64, ranks = intArrayOf(64, 40))
    }

    /** Above the LAPACK gate, so the host path is the one under test rather than the portable fallback. */
    @Test
    fun `factorInto refactorizes into the destination it was given`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertFactorIntoUsesItsDestination(F64Lapacke(), n = 96)
    }

    @Test
    fun `the factorizations match reference at blocked sizes`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertLuAgreesWithReference(F64Lapacke(), intArrayOf(7, 64, 256))
    }

    @Test
    fun `the level 2 and 3 products match reference at moderate sizes`() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
        val host = F64Cblas()
        assertGemvAgreesWithReference(host, intArrayOf(18, 64))
        assertGemmAgreesWithReference(host, intArrayOf(18, 64))
        assertSyrkAgreesWithReference(host, intArrayOf(18, 64))
        assertSyrkTriangleModesLeaveTheOtherTriangle(host, intArrayOf(18, 64))
        assertSymmetricProductsAgreeWithReference(host, intArrayOf(18, 64))
    }

    /** The same properties at sizes under every gate, where the portable fallback answers instead. */
    @Test
    fun `the level 2 and 3 products match reference at small sizes`() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
        val host = F64Cblas()
        assertGemvAgreesWithReference(host, intArrayOf(7))
        assertGemmAgreesWithReference(host, intArrayOf(6))
        assertSyrkAgreesWithReference(host, intArrayOf(6))
        assertSyrkTriangleModesLeaveTheOtherTriangle(host, intArrayOf(6))
        assertSymmetricProductsAgreeWithReference(host, intArrayOf(1, 6, 13))
    }

    @Test
    fun `degenerate shapes honor the beta conventions`() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
        assertDegenerateShapesHonorTheBetaConventions(F64Cblas())
    }

    /** Above the LAPACK gate of 64, so the host factorizations are the ones under test. */
    @Test
    fun `the factorization family matches reference at moderate sizes`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        val host = F64Lapacke()
        assertDeterminantAgreesWithReference(host, intArrayOf(64, 96))
        assertLuFactorsInterchange(host, n = 96)
        assertRcondAgreesWithReference(host, intArrayOf(64, 96))
        assertLdlBlockSolveAgreesWithReference(host, n = 96, nrhs = 4)
        assertLdlFactorsInterchange(host, intArrayOf(64, 96))
        assertQrFactorsInterchange(host, listOf(96 to 96, 128 to 64, 512 to 16))
    }

    /** The same properties at sizes under the gate, where the portable fallback answers instead. */
    @Test
    fun `the factorization family matches reference at small sizes`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        val host = F64Lapacke()
        assertDeterminantAgreesWithReference(host, intArrayOf(1, 3, 8, 33))
        assertLuFactorsInterchange(host, n = 12)
        assertRcondAgreesWithReference(host, intArrayOf(1, 6, 24))
        assertLdlBlockSolveAgreesWithReference(host, n = 9, nrhs = 4)
        assertLdlFactorsInterchange(host, intArrayOf(1, 2, 5, 14, 33))
        assertQrFactorsInterchange(host, listOf(6 to 6, 10 to 4))
    }

    @Test
    fun `a singular matrix sets the flag and a zero determinant`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertSingularLuIsFlagged(F64Lapacke())
    }

    @Test
    fun `an empty factorization solves an empty right-hand side`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        assertAnEmptyFactorizationSolvesEmpty(F64Lapacke())
    }

    /**
     * The level-1 gate is [Int.MAX_VALUE] while the compiled-in kernels are SIMD, so nothing otherwise calls
     * OpenBLAS's ddot, dnrm2, dasum, daxpy or dscal. Configured to route from length zero, these are the
     * only exercise those five downcalls get.
     */
    @Test
    fun `the host level-1 kernels agree with the compiled-in ones`() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
        val host = F64CblasKernels(HostBlasConfig())
        Assume.assumeTrue("host CBLAS did not bind its level-1 symbols", host.isAvailable)
        assertLevel1KernelsAgreeWithScalar(host)
        assertReductionsAgreeWithScalar(host)
        assertSwapAgreesWithScalar(host)
    }
}
