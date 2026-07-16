package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LinearAlgebraTest {

    @Test
    fun `koblas resolves to the reference backend when no native one is present`() {
        assertEquals(platformLinearAlgebra() ?: ReferenceLinearAlgebra, koblas)
    }

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
            val a = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-5.0, 5.0) })
            for (i in 0 until n) a[i, i] = a[i, i] + n * 10.0 // diagonally dominant ⇒ well-conditioned
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            val b = koblas.gemv(a, x)
            val solved = a.lu().solve(b)
            for (i in 0 until n) assertTrue(abs(solved[i] - x[i]) < 1e-9, "component $i: ${solved[i]} != ${x[i]}")
        }
    }

    @Test
    fun `LU transpose-solve solves the transposed system for BTRAN`() {
        val rng = Random(4242)
        repeat(100) {
            val n = rng.nextInt(1, 8)
            val a = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-5.0, 5.0) })
            for (i in 0 until n) a[i, i] = a[i, i] + n * 10.0
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            val b = koblas.gemv(a, x, transpose = true) // b = Aᵀ x
            val solved = a.lu().solve(b, transpose = true) // solve Aᵀ y = b ⇒ y == x
            for (i in 0 until n) assertTrue(abs(solved[i] - x[i]) < 1e-9, "component $i: ${solved[i]} != ${x[i]}")
        }
    }

    @Test
    fun `factor flags a singular matrix`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0))) // rank 1
        assertTrue(a.lu().singular)
    }
}
