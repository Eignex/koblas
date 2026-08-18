package com.eignex.koblas.hostblas

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.HostLibraryTest
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.dense.assertAnEmptyFactorizationSolvesEmpty
import com.eignex.koblas.dense.assertDegenerateShapesHonorTheBetaConventions
import com.eignex.koblas.dense.assertDeterminantAgreesWithReference
import com.eignex.koblas.dense.assertFactorIntoUsesItsDestination
import com.eignex.koblas.dense.assertGemmAgreesWithReference
import com.eignex.koblas.dense.assertGemvAgreesWithReference
import com.eignex.koblas.dense.assertGerAgreesWithReference
import com.eignex.koblas.dense.assertLdlBlockSolveAgreesWithReference
import com.eignex.koblas.dense.assertLdlFactorsInterchange
import com.eignex.koblas.dense.assertLevel3AgreesWithReference
import com.eignex.koblas.dense.assertLuAgreesWithReference
import com.eignex.koblas.dense.assertLuFactorsInterchange
import com.eignex.koblas.dense.assertNonPositiveDefiniteFallsBack
import com.eignex.koblas.dense.assertPivotedQrAgreesWithReference
import com.eignex.koblas.dense.assertQrFactorsInterchange
import com.eignex.koblas.dense.assertRcondAgreesWithReference
import com.eignex.koblas.dense.assertSingularLdlIsRefused
import com.eignex.koblas.dense.assertSingularLuIsFlagged
import com.eignex.koblas.dense.assertSpdSuiteAgreesWithReference
import com.eignex.koblas.dense.assertSymmetricProductsAgreeWithReference
import com.eignex.koblas.dense.assertSymvRefusesNonSquare
import com.eignex.koblas.dense.assertSyrkAgreesWithReference
import com.eignex.koblas.dense.assertSyrkTriangleModesLeaveTheOtherTriangle
import com.eignex.koblas.dense.assertTriangularAgreesWithReference
import com.eignex.koblas.koblasInfo
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.wellConditioned
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

@Category(HostLibraryTest::class)
class HostBlasConformanceTest {

    @Test
    fun `a full-uplo syrk returns its scratch buffer to the workspace`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        val n = 64
        val rng = Random(20260816)
        val a = randomMatrix(n, n, rng)
        val c = F64DenseMatrix.zero(n, n)
        val ws = Workspace()
        val host: F64Blas = HostBlas()
        // Prime the pool with the one buffer of this width, so syrk has to borrow and return that instance.
        val scratch = ws.take(n * n)
        ws.release(scratch)
        host.syrk(1.0, a, transpose = false, beta = 0.0, c = c, uplo = Uplo.FULL, workspace = ws)
        assertSame(scratch, ws.take(n * n), "syrk kept the scratch buffer instead of releasing it")
    }

    @Test
    fun `the host backend resolves when the machine has OpenBLAS`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        assertTrue(koblasInfo.contains("openblas"), koblasInfo)
    }

    /** A factorization at a size that engages OpenBLAS's parallel path. */
    @Test
    fun `a large factorization does not take the process down`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        val n = 512
        val rng = Random(20260807)
        val a = wellConditioned(n, rng)
        val host = HostLapack()
        repeat(4) {
            val lu = host.factor(a)
            assertEquals(n, lu.n)
        }
    }

    /** The level-2 half runs portable here whatever the size; the no-SIMD host pass is what reaches it. */
    @Test
    fun `the gated level 2 and 3 routines match reference`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        assertTriangularAgreesWithReference(HostBlas(), intArrayOf(1, 5, 12, 24))
        assertGerAgreesWithReference(HostBlas())
    }

    @Test
    fun `symv refuses a non-square matrix`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        assertSymvRefusesNonSquare(HostBlas())
    }

    @Test
    fun `a singular LDL is refused at every width`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        assertSingularLdlIsRefused(HostLapack())
    }

    @Test
    fun `level 3 matches reference at blocked sizes`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        assertLevel3AgreesWithReference(HostBlas(), intArrayOf(7, 64, 256))
    }

    /** Above cholesky's gate of 32 and invertSpd's of 16, so the host path is the one under test. */
    @Test
    fun `the SPD suite matches reference where the gates open`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        assertSpdSuiteAgreesWithReference(HostLapack(), intArrayOf(33, 256))
    }

    @Test
    fun `a non positive definite input falls back to the portable path`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        assertNonPositiveDefiniteFallsBack(HostLapack(), n = 256)
    }

    @Test
    fun `pivoted QR matches reference in rank and reconstruction`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        assertPivotedQrAgreesWithReference(HostLapack(), m = 96, cols = 64, ranks = intArrayOf(64, 40))
    }

    /** Above the LAPACK gate, so the host path is the one under test rather than the portable fallback. */
    @Test
    fun `factorInto refactorizes into the destination it was given`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        assertFactorIntoUsesItsDestination(HostLapack(), n = 96)
    }

    @Test
    fun `the factorizations match reference at blocked sizes`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        assertLuAgreesWithReference(HostLapack(), intArrayOf(7, 64, 256))
    }

    /** Both extents stay at or above the level-3 gate of 16, so the host kernels are the ones under test. */
    @Test
    fun `the level 2 and 3 products match reference above the gates`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        val host = HostBlas()
        assertGemvAgreesWithReference(host, intArrayOf(18, 64))
        assertGemmAgreesWithReference(host, intArrayOf(18, 64))
        assertSyrkAgreesWithReference(host, intArrayOf(18, 64))
        assertSyrkTriangleModesLeaveTheOtherTriangle(host, intArrayOf(18, 64))
        assertSymmetricProductsAgreeWithReference(host, intArrayOf(18, 64))
    }

    /** The same properties at sizes under every gate, where the portable fallback answers instead. */
    @Test
    fun `the level 2 and 3 products match reference below the gates`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        val host = HostBlas()
        assertGemvAgreesWithReference(host, intArrayOf(7))
        assertGemmAgreesWithReference(host, intArrayOf(6))
        assertSyrkAgreesWithReference(host, intArrayOf(6))
        assertSyrkTriangleModesLeaveTheOtherTriangle(host, intArrayOf(6))
        assertSymmetricProductsAgreeWithReference(host, intArrayOf(1, 6, 13))
    }

    @Test
    fun `degenerate shapes honor the beta conventions`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        assertDegenerateShapesHonorTheBetaConventions(HostBlas())
    }

    /** Above the LAPACK gate of 64, so the host factorizations are the ones under test. */
    @Test
    fun `the factorization family matches reference above the gate`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        val host = HostLapack()
        assertDeterminantAgreesWithReference(host, intArrayOf(64, 96))
        assertLuFactorsInterchange(host, n = 96)
        assertRcondAgreesWithReference(host, intArrayOf(64, 96))
        assertLdlBlockSolveAgreesWithReference(host, n = 96, nrhs = 4)
        assertLdlFactorsInterchange(host, intArrayOf(64, 96))
        assertQrFactorsInterchange(host, listOf(96 to 96, 128 to 64))
    }

    /** The same properties at sizes under the gate, where the portable fallback answers instead. */
    @Test
    fun `the factorization family matches reference below the gate`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        val host = HostLapack()
        assertDeterminantAgreesWithReference(host, intArrayOf(1, 3, 8, 33))
        assertLuFactorsInterchange(host, n = 12)
        assertRcondAgreesWithReference(host, intArrayOf(1, 6, 24))
        assertLdlBlockSolveAgreesWithReference(host, n = 9, nrhs = 4)
        assertLdlFactorsInterchange(host, intArrayOf(1, 2, 5, 14, 33))
        assertQrFactorsInterchange(host, listOf(6 to 6, 10 to 4))
    }

    @Test
    fun `a singular matrix sets the flag and a zero determinant`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        assertSingularLuIsFlagged(HostLapack())
    }

    @Test
    fun `an empty factorization solves an empty right-hand side`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        assertAnEmptyFactorizationSolvesEmpty(HostLapack())
    }
}
