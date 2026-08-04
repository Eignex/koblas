package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.wellConditioned
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The general dense routines: `gemv`, `gemm`, and the LU factorization with its solve in both directions.
 *
 * Small products are checked against hand-computed values, and the solves against randomly generated
 * systems with a known answer, which reaches the pivoting paths a fixed example would miss. The degenerate
 * and rejected shapes are here too: an empty or 1x1 matrix exercises loop bounds that never run in the
 * general case, and the argument checks are part of the contract callers rely on.
 */
class LinearAlgebraTest {

    @Test
    fun `gemv multiplies a matrix and its transpose by a vector`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0))) // 2x3
        assertTrue(doubleArrayOf(14.0, 32.0).contentEquals(koblas.gemv(a, doubleArrayOf(1.0, 2.0, 3.0))))
        assertTrue(
            doubleArrayOf(9.0, 12.0, 15.0).contentEquals(koblas.gemv(a, doubleArrayOf(1.0, 2.0), transpose = true)),
        )
    }

    @Test
    fun `gemm multiplies two matrices`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val b = DenseMatrix.of(arrayOf(doubleArrayOf(5.0, 6.0), doubleArrayOf(7.0, 8.0)))
        assertEquals(DenseMatrix.of(arrayOf(doubleArrayOf(19.0, 22.0), doubleArrayOf(43.0, 50.0))), a.matMul(b))
    }

    @Test
    fun `LU factor and solve recover the solution of a random system`() {
        val rng = Random(20260716)
        repeat(200) {
            val n = rng.nextInt(1, 9)
            val a = wellConditioned(n, rng)
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            val b = koblas.gemv(a, x)
            assertClose(x, a.lu().solve(b), "solve n=$n", tolerance = 1e-9)
        }
    }

    @Test
    fun `LU transpose-solve solves the transposed system for BTRAN`() {
        val rng = Random(4242)
        repeat(100) {
            val n = rng.nextInt(1, 8)
            val a = wellConditioned(n, rng)
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            val b = koblas.gemv(a, x, transpose = true) // b = Aᵀ x
            // Solve Aᵀ y = b, so y == x.
            assertClose(x, a.lu().solve(b, transpose = true), "transpose-solve n=$n", tolerance = 1e-9)
        }
    }

    @Test
    fun `factor flags a singular matrix`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0))) // rank 1
        assertTrue(a.lu().singular)
        assertTrue(DenseMatrix.of(arrayOf(doubleArrayOf(0.0))).lu().singular)
    }

    @Test
    fun `1x1 solve and transpose-solve`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(4.0)))
        assertEquals(0.5, a.lu().solve(doubleArrayOf(2.0))[0], 1e-12)
        assertEquals(0.5, a.lu().solve(doubleArrayOf(2.0), transpose = true)[0], 1e-12)
    }

    @Test
    fun `gemm with zero inner dimension yields a zero matrix`() {
        val a = DenseMatrix(2, 0)
        val b = DenseMatrix(0, 3)
        assertEquals(DenseMatrix(2, 3), a.matMul(b))
    }

    @Test
    fun `the dense routines reject mismatched shapes`() {
        assertFailsWith<IllegalArgumentException> { DenseMatrix(2, 3).lu() }
        assertFailsWith<IllegalArgumentException> { DenseMatrix(2, 3).matMul(DenseMatrix(2, 2)) }
        assertFailsWith<IllegalArgumentException> { koblas.gemv(DenseMatrix(2, 3), DoubleArray(2)) }
        assertFailsWith<IllegalArgumentException> { koblas.gemv(DenseMatrix(2, 3), DoubleArray(3), transpose = true) }
        assertFailsWith<IllegalArgumentException> { DenseMatrix.diagonal(3).lu().solve(DoubleArray(2)) }
    }
}
