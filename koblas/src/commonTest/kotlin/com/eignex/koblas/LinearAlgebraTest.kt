package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LinearAlgebraTest {

    private val la = ReferenceLinearAlgebra

    @Test
    fun `koblas resolves to the reference backend when no native one is present`() {
        // platformLinearAlgebra() is null on every target today, so the default is the reference.
        assertEquals(platformLinearAlgebra() ?: ReferenceLinearAlgebra, koblas)
    }

    @Test
    fun `dot axpy and scal compute the elementwise results`() {
        assertEquals(32.0, la.dot(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)), 1e-12)
        val y = doubleArrayOf(1.0, 1.0, 1.0)
        la.axpy(2.0, doubleArrayOf(3.0, 4.0, 5.0), y)
        assertTrue(doubleArrayOf(7.0, 9.0, 11.0).contentEquals(y))
        val x = doubleArrayOf(1.0, -2.0, 3.0)
        la.scal(0.5, x)
        assertTrue(doubleArrayOf(0.5, -1.0, 1.5).contentEquals(x))
    }

    @Test
    fun `gemv multiplies a matrix and its transpose by a vector`() {
        val a = Matrix.ofRows(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)) // 2x3
        assertTrue(doubleArrayOf(14.0, 32.0).contentEquals(la.gemv(a, doubleArrayOf(1.0, 2.0, 3.0))))
        assertTrue(doubleArrayOf(9.0, 12.0, 15.0).contentEquals(la.gemv(a, doubleArrayOf(1.0, 2.0), transpose = true)))
    }

    @Test
    fun `gemm multiplies two matrices`() {
        val a = Matrix.ofRows(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val b = Matrix.ofRows(doubleArrayOf(5.0, 6.0), doubleArrayOf(7.0, 8.0))
        assertEquals(Matrix.ofRows(doubleArrayOf(19.0, 22.0), doubleArrayOf(43.0, 50.0)), la.gemm(a, b))
    }

    @Test
    fun `LU factor and solve recover the solution of a random system`() {
        val rng = Random(20260716)
        repeat(200) {
            val n = rng.nextInt(1, 9)
            val a = Matrix(n, n, DoubleArray(n * n) { rng.nextDouble(-5.0, 5.0) })
            // Diagonally dominant so the matrix is non-singular and well-conditioned.
            for (i in 0 until n) a[i, i] = a[i, i] + n * 10.0
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            val b = la.gemv(a, x)
            val solved = la.solve(la.factor(a), b)
            for (i in 0 until n) assertTrue(abs(solved[i] - x[i]) < 1e-9, "component $i: ${solved[i]} != ${x[i]}")
        }
    }

    @Test
    fun `factor flags a singular matrix`() {
        val a = Matrix.ofRows(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)) // rank 1
        assertTrue(la.factor(a).singular)
    }
}
