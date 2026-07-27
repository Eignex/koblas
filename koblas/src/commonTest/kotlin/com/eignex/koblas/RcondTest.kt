package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RcondTest {

    @Test
    fun `identity has rcond one and norm1 one`() {
        val a = DenseMatrix.diagonal(5)
        assertEquals(1.0, norm1(a), 1e-15)
        assertEquals(1.0, koblas.rcond(a.lu(), norm1(a)), 1e-12)
    }

    @Test
    fun `diagonal matrix estimate is exact`() {
        // diag(1, 1e-6): norm1 = 1, inverse norm1 = 1e6, so rcond = 1e-6.
        val a = DenseMatrix(2)
        a[0, 0] = 1.0
        a[1, 1] = 1e-6
        val rc = koblas.rcond(a.lu(), norm1(a))
        assertTrue(abs(rc - 1e-6) <= 1e-9 * 1e-6 + 1e-18, "rcond $rc != 1e-6")
    }

    @Test
    fun `singular and degenerate conventions`() {
        val singular = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        assertEquals(0.0, koblas.rcond(singular.lu(), norm1(singular)))
        val regular = DenseMatrix.diagonal(2)
        assertEquals(0.0, koblas.rcond(regular.lu(), 0.0))
        assertEquals(1.0, koblas.rcond(DenseMatrix(0, 0).lu(), 0.0))
    }

    @Test
    fun `estimate never understates and stays within a magnitude of the truth`() {
        val rng = Random(20260901)
        for (n in intArrayOf(3, 8, 20)) {
            val a = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            for (i in 0 until n) a[i, i] = a[i, i] + n
            val lu = a.lu()
            // Exact inverse 1-norm: solve every unit vector and take the max column sum.
            var exactInvNorm = 0.0
            for (j in 0 until n) {
                val e = DoubleArray(n)
                e[j] = 1.0
                val col = koblas.solve(lu, e)
                var s = 0.0
                for (v in col) s += abs(v)
                if (s > exactInvNorm) exactInvNorm = s
            }
            val exact = 1.0 / (norm1(a) * exactInvNorm)
            val estimate = koblas.rcond(lu, norm1(a))
            assertTrue(estimate >= exact * (1 - 1e-12), "n=$n: estimate $estimate below exact $exact")
            assertTrue(estimate <= exact * 10.0, "n=$n: estimate $estimate above 10x exact $exact")
        }
    }

    @Test
    fun `rcond is invariant under matrix scaling`() {
        val rng = Random(20260902)
        val n = 6
        val a = DenseMatrix(n)
        for (i in 0 until n) for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0) + if (i == j) n.toDouble() else 0.0
        val scaled = DenseMatrix(n)
        for (i in 0 until n) for (j in 0 until n) scaled[i, j] = 8.0 * a[i, j]
        val rc = koblas.rcond(a.lu(), norm1(a))
        val rcScaled = koblas.rcond(scaled.lu(), norm1(scaled))
        assertEquals(rc, rcScaled, 1e-12 * rc)
    }

    @Test
    fun `norm1 is the maximum absolute column sum`() {
        val a = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, -2.0, 3.0),
                doubleArrayOf(-4.0, 5.0, -6.0),
            ),
        )
        assertEquals(9.0, norm1(a)) // columns sum to 5, 7, 9
        assertEquals(0.0, norm1(DenseMatrix(0, 3)))
        assertEquals(0.0, norm1(DenseMatrix(3, 0)))
    }
}
