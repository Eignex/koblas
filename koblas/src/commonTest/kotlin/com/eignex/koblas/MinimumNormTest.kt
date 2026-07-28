package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class MinimumNormTest {

    private fun assertClose(expected: DoubleArray, actual: DoubleArray, tol: Double, context: String) {
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= tol * maxOf(1.0, abs(expected[i])),
                "$context index $i: expected ${expected[i]} actual ${actual[i]}",
            )
        }
    }

    private fun transposed(a: DenseMatrix): DenseMatrix {
        val t = DenseMatrix(a.cols, a.rows)
        for (i in 0 until a.rows) for (j in 0 until a.cols) t[j, i] = a[i, j]
        return t
    }

    @Test
    fun `solves wide systems exactly with the smallest norm`() {
        val rng = Random(20260960)
        for ((m, n) in listOf(2 to 4, 3 to 7, 5 to 6)) {
            val a = DenseMatrix(m, n)
            for (i in 0 until m) for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0)
            val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
            val x = koblas.solveMinimumNorm(koblas.qr(transposed(a)), b)
            // Exact solution: A · x = b.
            val ax = koblas.gemv(a, x)
            assertClose(b, ax, 1e-11, "residual ${m}x$n")
            // Minimum norm: x equals the pseudoinverse solution Aᵀ·(A·Aᵀ)⁻¹·b.
            val g = DenseMatrix(m, m)
            koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = g)
            val z = koblas.solve(koblas.factor(g), b)
            val expected = koblas.gemv(a, z, transpose = true)
            assertClose(expected, x, 1e-10, "min-norm ${m}x$n")
        }
    }

    @Test
    fun `square systems agree with the LU solve`() {
        val rng = Random(20260961)
        val n = 6
        val a = DenseMatrix(n)
        for (i in 0 until n) for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0) + if (i == j) n.toDouble() else 0.0
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val viaLu = koblas.solve(koblas.factor(a), b)
        val viaMinNorm = koblas.solveMinimumNorm(koblas.qr(transposed(a)), b)
        assertClose(viaLu, viaMinNorm, 1e-10, "square")
    }

    @Test
    fun `rejects a wide factorization`() {
        // The factorization must be of Aᵀ (tall); passing the wide QR of A itself is a misuse.
        val a = DenseMatrix(2, 5)
        for (i in 0 until 2) for (j in 0 until 5) a[i, j] = (i + j + 1).toDouble()
        assertFailsWith<IllegalArgumentException> {
            koblas.solveMinimumNorm(koblas.qr(a), DoubleArray(5))
        }
    }
}
