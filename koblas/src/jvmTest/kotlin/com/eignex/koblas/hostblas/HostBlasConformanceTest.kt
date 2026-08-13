package com.eignex.koblas.hostblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HostLibraryTest
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.assertGerAgreesWithReference
import com.eignex.koblas.dense.assertLevel3AgreesWithReference
import com.eignex.koblas.dense.assertLuAgreesWithReference
import com.eignex.koblas.dense.assertNonPositiveDefiniteFallsBack
import com.eignex.koblas.dense.assertSingularLdlIsRefused
import com.eignex.koblas.dense.assertSpdSuiteAgreesWithReference
import com.eignex.koblas.dense.assertSymvRefusesNonSquare
import com.eignex.koblas.dense.assertTriangularAgreesWithReference
import com.eignex.koblas.koblasInfo
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@Category(HostLibraryTest::class)
class HostBlasConformanceTest {

    private fun randomMatrix(rng: Random, rows: Int, cols: Int) =
        DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

    private fun assertClose(expected: DoubleArray, actual: DoubleArray, tol: Double, context: String) {
        assertEquals(expected.size, actual.size, context)
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= tol * maxOf(1.0, abs(expected[i])),
                "$context index $i: expected ${expected[i]} actual ${actual[i]}",
            )
        }
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
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        for (i in 0 until n) a[i, i] = a[i, i] + n
        val host = HostLapack()
        repeat(4) {
            val lu = host.factor(a)
            assertEquals(n, lu.n)
        }
    }

    @Test
    fun `the gated level 2 and 3 routines match reference when their gates are opened`() {
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
        val host = HostLapack()
        val rng = Random(20260809)
        val m = 96
        for (rank in intArrayOf(64, 40)) {
            // A rank-deficient product of full-rank factors, where 64 is full column rank and 40 is deficient.
            val left = randomMatrix(rng, m, rank)
            val right = randomMatrix(rng, rank, 64)
            val a = DenseMatrix(m, 64)
            ReferenceLinearAlgebra.gemm(1.0, left, false, right, false, 0.0, a)

            val expected = ReferenceLinearAlgebra.qrPivoted(a)
            val actual = host.qrPivoted(a)
            assertEquals(expected.rank, actual.rank, "rank of a ${m}x64 matrix built at rank $rank")
            assertEquals((0 until 64).toList(), actual.pivots.sorted(), "pivots must be a permutation")

            for (j in 0 until 64) {
                val rColumn = DoubleArray(m)
                for (i in 0..minOf(j, actual.factorization.tau.size - 1)) {
                    rColumn[i] = actual.factorization.qr[i + j * m]
                }
                val rebuilt = host.applyQ(actual.factorization, rColumn)
                val original = DoubleArray(m) { i -> a[i, actual.pivots[j]] }
                assertClose(original, rebuilt, tol = 1e-8, context = "A·P = Q·R rank=$rank column $j")
            }

            val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
            assertClose(
                ReferenceLinearAlgebra.solveLeastSquares(expected, b),
                host.solveLeastSquares(actual, b),
                tol = 1e-7,
                context = "pivoted least squares rank=$rank",
            )
        }
    }

    @Test
    fun `the factorizations match reference at blocked sizes`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        assertLuAgreesWithReference(HostLapack(), intArrayOf(7, 64, 256))
    }
}
