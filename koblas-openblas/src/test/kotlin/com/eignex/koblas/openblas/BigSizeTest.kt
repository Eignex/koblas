package com.eignex.koblas.openblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ReferenceLinearAlgebra
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The conformance suite runs at tiny sizes, where OpenBLAS stays on its serial path. Sizes like these are
 * where it would otherwise spawn threads — whose parallel LAPACK overflows default JVM thread stacks and
 * takes the process down — so this is the regression guard for the backend's threading configuration as
 * much as for its arithmetic.
 */
class BigSizeTest {
    @Test
    fun `level 3 and factorization agree with the reference at realistic sizes`() {
        val openblas = OpenBlasLinearAlgebra()
        val rng = Random(20260729)
        for (n in intArrayOf(64, 256)) {
            val a = DenseMatrix(n, n)
            for (i in 0 until n) {
                for (j in 0 until n) {
                    a[i, j] = rng.nextDouble(
                        -1.0,
                        1.0,
                    ) + if (i == j) n.toDouble() else 0.0
                }
            }
            val b = DenseMatrix(n, n)
            for (i in 0 until n) for (j in 0 until n) b[i, j] = rng.nextDouble(-1.0, 1.0)
            val expected = ReferenceLinearAlgebra.gemm(a, b)
            val actual = openblas.gemm(a, b)
            var worst = 0.0
            for (i in expected.data.indices) worst = maxOf(worst, abs(expected.data[i] - actual.data[i]))
            assertTrue(worst < 1e-9 * n, "gemm n=$n worst diff $worst")
            val rhs = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val x = openblas.solve(openblas.factor(a), rhs)
            val xr = ReferenceLinearAlgebra.solve(ReferenceLinearAlgebra.factor(a), rhs)
            for (i in 0 until n) assertTrue(abs(x[i] - xr[i]) < 1e-9, "solve n=$n at $i: ${x[i]} vs ${xr[i]}")
        }
    }
}
