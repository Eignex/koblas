package com.eignex.koblas.hostblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HostLibraryTest
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.koblasInfo
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The JVM host-BLAS backend against the portable reference.
 *
 * Skips itself when the machine has no OpenBLAS, which is a legitimate configuration rather than a
 * failure: resolving by soname is what makes the library optional. CI installs it, so the assertions do
 * run there.
 *
 * Sizes reach 256 deliberately. Below roughly 50 OpenBLAS stays on its serial path and its blocked
 * kernels never engage, so a suite that stopped short would leave the code that matters untested — a gap
 * that once let a threading crash reach a benchmark run rather than a test.
 */
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

    /**
     * A factorization at a size that engages OpenBLAS's parallel path.
     *
     * Unconfigured, that path overflows a default JVM thread stack and takes the process down with
     * SIGSEGV — no exception, nothing to catch, and every test in the run reported as passing before the
     * crash. That is exactly how it reached CI once: the threading setup was dropped when this backend was
     * assembled. This runs the shape that crashes, so a regression fails here instead of in someone's
     * application.
     */
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

    /**
     * The routines whose gates are shut by default, exercised anyway.
     *
     * `ger`, `trsv`, `trmv`, `trsm` and `trmm` stay portable at the shipped thresholds, so nothing else in
     * the suite would ever execute their native paths — the bindings would be dead code that compiles.
     * Overriding the gates through the documented properties runs them, which is also a check that the
     * override plumbing works.
     */
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
