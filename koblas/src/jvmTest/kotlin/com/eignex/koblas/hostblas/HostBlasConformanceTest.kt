package com.eignex.koblas.hostblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ReferenceLinearAlgebra
import com.eignex.koblas.Uplo
import com.eignex.koblas.koblasInfo
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
class HostBlasConformanceTest {

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
        if (!HostBlasCalls.blasAvailable) return
        // Either the backend's own name or a composed one, depending on whether LAPACKE resolved too.
        assertTrue(koblasInfo.contains("openblas"), koblasInfo)
    }

    @Test
    fun `level 3 matches reference at blocked sizes`() {
        if (!HostBlasCalls.blasAvailable) return
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
        if (!HostBlasCalls.lapackAvailable) return
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
