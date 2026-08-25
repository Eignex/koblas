package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

class LinearAlgebraRcondTest {

    @Test
    fun `identity has rcond one and norm1 one`() {
        val a = F64DenseMatrix.diagonal(5)
        assertEquals(1.0, a.norm1(), 1e-15)
        assertEquals(1.0, koblas.rcond(a.lu(), a.norm1()), 1e-12)
    }

    @Test
    fun `diagonal matrix estimate is exact`() {
        // For diag(1, 1e-6) the 1-norm is 1 and the inverse 1-norm is 1e6, so rcond is 1e-6.
        val a = F64DenseMatrix(2)
        a[0, 0] = 1.0
        a[1, 1] = 1e-6
        val rc = koblas.rcond(a.lu(), a.norm1())
        assertTrue(abs(rc - 1e-6) <= 1e-9 * 1e-6 + 1e-18, "rcond $rc != 1e-6")
    }

    @Test
    fun `singular and degenerate conventions`() {
        val singular = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        assertEquals(0.0, koblas.rcond(singular.lu(), singular.norm1()))
        val regular = F64DenseMatrix.diagonal(2)
        assertEquals(0.0, koblas.rcond(regular.lu(), 0.0))
        assertEquals(1.0, koblas.rcond(F64DenseMatrix(0, 0).lu(), 0.0))
    }

    @Test
    fun `estimate never understates and stays within a magnitude of the truth`() {
        val rng = Random(20260901)
        for (n in intArrayOf(3, 8, 20)) {
            val a = wellConditioned(n, rng)
            val lu = a.lu()
            // The exact inverse 1-norm comes from solving every unit vector and taking the largest column sum.
            var exactInvNorm = 0.0
            for (j in 0 until n) {
                val e = DoubleArray(n)
                e[j] = 1.0
                val col = koblas.solve(lu, e)
                var s = 0.0
                for (v in col) s += abs(v)
                if (s > exactInvNorm) exactInvNorm = s
            }
            val exact = 1.0 / (a.norm1() * exactInvNorm)
            val estimate = koblas.rcond(lu, a.norm1())
            assertTrue(estimate >= exact * (1 - 1e-12), "n=$n: estimate $estimate below exact $exact")
            assertTrue(estimate <= exact * 10.0, "n=$n: estimate $estimate above 10x exact $exact")
        }
    }

    @Test
    fun `rcond is invariant under matrix scaling`() {
        val rng = Random(20260902)
        val n = 6
        val a = wellConditioned(n, rng)
        val scaled = F64DenseMatrix(n)
        for (i in 0 until n) for (j in 0 until n) scaled[i, j] = 8.0 * a[i, j]
        val rc = koblas.rcond(a.lu(), a.norm1())
        val rcScaled = koblas.rcond(scaled.lu(), scaled.norm1())
        assertEquals(rc, rcScaled, 1e-12 * rc)
    }
}
