package com.eignex.koblas.hostblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HostLibraryTest
import com.eignex.koblas.dense.CholeskyPolicy
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.koblasInfo
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** The JVM host-BLAS backend against the portable reference. */
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
        // Either the backend's own name or a composed one, depending on whether LAPACKE resolved too.
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

    /** The routines whose gates are shut by default, exercised anyway. */
    @Test
    fun `the gated level 2 and 3 routines match reference when their gates are opened`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        val host = HostBlas()
        val rng = Random(20260808)
        val n = 24
        val nrhs = 5
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val t = DenseMatrix(n, n)
                    for (i in 0 until n) {
                        for (j in 0 until n) {
                            val strict = if (lower) j < i else j > i
                            t[i, j] = when {
                                strict -> rng.nextDouble(-1.0, 1.0)
                                i == j -> rng.nextDouble(2.0, 4.0)
                                else -> 0.0
                            }
                        }
                    }
                    val flags = "lower=$lower t=$transpose unit=$unitDiag"
                    val x = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
                    val portableTrsv = x.copyOf().also {
                        ReferenceLinearAlgebra.trsv(t, it, lower, transpose, unitDiag)
                    }
                    val nativeTrsv = x.copyOf().also { host.trsv(t, it, lower, transpose, unitDiag) }
                    assertClose(portableTrsv, nativeTrsv, tol = 1e-9, context = "trsv $flags")
                    val portableTrmv = x.copyOf().also {
                        ReferenceLinearAlgebra.trmv(t, it, lower, transpose, unitDiag)
                    }
                    val nativeTrmv = x.copyOf().also { host.trmv(t, it, lower, transpose, unitDiag) }
                    assertClose(portableTrmv, nativeTrmv, tol = 1e-9, context = "trmv $flags")
                    for (right in booleanArrayOf(false, true)) {
                        val b = if (right) randomMatrix(rng, nrhs, n) else randomMatrix(rng, n, nrhs)
                        for (solve in booleanArrayOf(true, false)) {
                            val expected = DenseMatrix(b.rows, b.cols, b.data.copyOf())
                            val actual = DenseMatrix(b.rows, b.cols, b.data.copyOf())
                            if (solve) {
                                ReferenceLinearAlgebra.trsm(t, expected, lower, transpose, unitDiag, right)
                                host.trsm(t, actual, lower, transpose, unitDiag, right)
                            } else {
                                ReferenceLinearAlgebra.trmm(t, expected, lower, transpose, unitDiag, right)
                                host.trmm(t, actual, lower, transpose, unitDiag, right)
                            }
                            val label = if (solve) "trsm" else "trmm"
                            assertClose(expected.data, actual.data, tol = 1e-9, context = "$label right=$right $flags")
                        }
                    }
                }
            }
        }
        // ger, over shapes and the alpha BLAS treats specially.
        for ((rows, cols) in listOf(24 to 24, 30 to 9)) {
            for (alpha in doubleArrayOf(0.0, -0.75)) {
                val xv = DoubleArray(rows) { rng.nextDouble(-1.0, 1.0) }
                val yv = DoubleArray(cols) { rng.nextDouble(-1.0, 1.0) }
                val a0 = randomMatrix(rng, rows, cols)
                val expected = DenseMatrix(rows, cols, a0.data.copyOf())
                val actual = DenseMatrix(rows, cols, a0.data.copyOf())
                ReferenceLinearAlgebra.ger(alpha, xv, yv, expected)
                host.ger(alpha, xv, yv, actual)
                assertClose(expected.data, actual.data, context = "ger ${rows}x$cols alpha=$alpha", tol = 1e-12)
            }
        }
    }

    @Test
    fun `level 3 matches reference at blocked sizes`() {
        Assume.assumeTrue("host CBLAS is not installed", HostBlasCalls.available)
        val host = HostBlas()
        val rng = Random(20260805)
        for (n in intArrayOf(7, 64, 256)) {
            val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            val b = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            assertClose(
                ReferenceLinearAlgebra.gemm(a, b).data,
                host.gemm(a, b).data,
                tol = 1e-9 * n,
                context = "gemm n=$n",
            )
            for (uplo in listOf(Uplo.FULL, Uplo.LOWER)) {
                val expected = DenseMatrix(n, n)
                val actual = DenseMatrix(n, n)
                ReferenceLinearAlgebra.syrk(1.0, a, transpose = true, beta = 0.0, c = expected, uplo = uplo)
                host.syrk(1.0, a, transpose = true, beta = 0.0, c = actual, uplo = uplo)
                assertClose(expected.data, actual.data, tol = 1e-9 * n, context = "syrk $uplo n=$n")
            }
        }
    }

    /** The SPD suite — `dpotrf`, `dpotrs`, `dpotri` — against the portable implementation. */
    @Test
    fun `the SPD suite matches reference where the gates open`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        val host = HostLapack()
        val rng = Random(20260807)
        // Comfortably above cholesky's gate of 32 and invertSpd's of 16, and small enough that the portable
        // comparison is not the slow part of the suite.
        val n = 256
        val a = spdMatrix(n, rng)
        for (i in 0 until n) for (j in i + 1 until n) a[i, j] = Double.NaN

        val expected = ReferenceLinearAlgebra.cholesky(a)
        val actual = host.cholesky(a)
        assertClose(expected.l.data, actual.l.data, tol = 1e-9, context = "cholesky n=$n")
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                assertEquals(0.0, actual.l[i, j], "cholesky n=$n left ${actual.l[i, j]} above the diagonal")
            }
        }

        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        assertClose(
            ReferenceLinearAlgebra.solve(expected, b),
            host.solve(actual, b),
            tol = 1e-9,
            context = "solve(chol) n=$n (portable by its gate, so this pins the fallback)",
        )

        val inverse = host.invert(actual)
        assertClose(
            ReferenceLinearAlgebra.invert(expected).data,
            inverse.data,
            tol = 1e-7,
            context = "invertSpd n=$n",
        )
        for (i in 0 until n step 37) {
            for (j in 0 until n step 41) {
                assertEquals(inverse[i, j], inverse[j, i], 1e-12, "invertSpd n=$n is not symmetric at ($i,$j)")
            }
        }
    }

    /**
     * A non-positive-definite input has no LAPACK equivalent: `dpotrf` reports the failing leading minor rather than
     * clamping it, so the host path hands the whole factorization back to the portable one. Both policies must
     * therefore behave identically on both backends — asserted at a size above the gate, since below it the host path
     * is not involved at all.
     */
    @Test
    fun `a non positive definite input falls back to the portable path`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        val host = HostLapack()
        val n = 256
        val bad = spdMatrix(n, Random(20260808))
        bad[n - 1, n - 1] = -1.0 // breaks positive definiteness at the last leading minor
        val policy = CholeskyPolicy.Regularize()
        assertEquals(
            ReferenceLinearAlgebra.cholesky(bad, policy).l[n - 1, n - 1],
            host.cholesky(bad, policy).l[n - 1, n - 1],
            1e-15,
        )
        assertFailsWith<IllegalArgumentException> { host.cholesky(bad) }
    }

    /** `dgeqp3` against the portable pivoted QR. */
    @Test
    fun `pivoted QR matches reference in rank and reconstruction`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        val host = HostLapack()
        val rng = Random(20260809)
        val m = 96
        for (rank in intArrayOf(64, 40)) {
            // A rank-deficient matrix as a product of full-rank factors, so the rank is known rather than
            // inferred: 64 is full column rank here, 40 is deficient.
            val left = randomMatrix(rng, m, rank)
            val right = randomMatrix(rng, rank, 64)
            val a = DenseMatrix(m, 64)
            ReferenceLinearAlgebra.gemm(1.0, left, false, right, false, 0.0, a)

            val expected = ReferenceLinearAlgebra.qrPivoted(a)
            val actual = host.qrPivoted(a)
            assertEquals(expected.rank, actual.rank, "rank of a ${m}x64 matrix built at rank $rank")
            assertEquals((0 until 64).toList(), actual.pivots.sorted(), "pivots must be a permutation")

            // A·P = Q·R, column by column, which is the only statement about P that holds for both.
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

    /** A symmetric positive-definite matrix: symmetrized noise with a dominant diagonal. */
    private fun spdMatrix(n: Int, rng: Random): DenseMatrix {
        val a = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0..i) {
                val v = rng.nextDouble(-1.0, 1.0)
                a[i, j] = v
                a[j, i] = v
            }
            a[i, i] = a[i, i] + n
        }
        return a
    }

    @Test
    fun `the factorizations match reference at blocked sizes`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        val host = HostLapack()
        val rng = Random(20260806)
        for (n in intArrayOf(7, 64, 256)) {
            val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            for (i in 0 until n) a[i, i] = a[i, i] + n
            val rhs = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            assertClose(
                ReferenceLinearAlgebra.solve(ReferenceLinearAlgebra.factor(a), rhs),
                host.solve(host.factor(a), rhs),
                tol = 1e-9,
                context = "LU solve n=$n",
            )
            // The blocked multi-RHS path, which is where the native trsm earns its call.
            val nrhs = 8
            val b = DenseMatrix.wrap(n, nrhs, DoubleArray(n * nrhs) { rng.nextDouble(-1.0, 1.0) })
            assertClose(
                ReferenceLinearAlgebra.solve(ReferenceLinearAlgebra.factor(a), b).data,
                host.solve(host.factor(a), b).data,
                tol = 1e-9,
                context = "LU block solve n=$n",
            )
        }
    }
}
