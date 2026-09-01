package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.random.Random
import kotlin.test.*

class LinearAlgebraTest {

    @Test
    fun `gemv multiplies a matrix and its transpose by a vector`() {
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0))) // 2x3
        assertTrue(doubleArrayOf(14.0, 32.0).contentEquals(koblas.gemv(a, doubleArrayOf(1.0, 2.0, 3.0))))
        assertTrue(
            doubleArrayOf(9.0, 12.0, 15.0).contentEquals(koblas.gemv(a, doubleArrayOf(1.0, 2.0), transpose = true)),
        )
    }

    @Test
    fun `gemm multiplies two matrices`() {
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val b = F64DenseMatrix.of(arrayOf(doubleArrayOf(5.0, 6.0), doubleArrayOf(7.0, 8.0)))
        assertEquals(F64DenseMatrix.of(arrayOf(doubleArrayOf(19.0, 22.0), doubleArrayOf(43.0, 50.0))), a * b)
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
            assertClose(x, a.lu().solve(b, transpose = true), "transpose-solve n=$n", tolerance = 1e-9)
        }
    }

    @Test
    fun `factor reports the first zero pivot position`() {
        assertEquals(NOT_SINGULAR, F64DenseMatrix.diagonal(3).lu().failedAt, "a good factorization has no position")
        val rank1 = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        assertEquals(1, rank1.lu().failedAt)
        assertEquals(0, F64DenseMatrix.of(arrayOf(doubleArrayOf(0.0))).lu().failedAt)
        assertEquals(0, F64DenseMatrix(3, 3).lu().failedAt, "the first of three zero pivots, not the last")
    }

    @Test
    fun `LU exposes a zero based defensive row permutation`() {
        val factor = F64DenseMatrix.of(arrayOf(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0, 0.0))).lu()

        assertContentEquals(intArrayOf(1, 0), factor.rowPermutation)
        assertEquals(1, factor.rowAt(0))
        val copy = factor.rowPermutation
        copy[0] = 0
        assertEquals(1, factor.rowAt(0), "changing the returned permutation must not alter the factor")
    }

    @Test
    fun `lapackFailedAt converts an info return`() {
        assertEquals(NOT_SINGULAR, lapackFailedAt(0))
        assertEquals(0, lapackFailedAt(1))
        assertEquals(4, lapackFailedAt(5))
        assertEquals(NOT_SINGULAR, lapackFailedAt(-3), "an illegal-argument report is not a singularity")
    }

    @Test
    fun `1x1 solve and transpose-solve`() {
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(4.0)))
        assertEquals(0.5, a.lu().solve(doubleArrayOf(2.0))[0], 1e-12)
        assertEquals(0.5, a.lu().solve(doubleArrayOf(2.0), transpose = true)[0], 1e-12)
    }

    @Test
    fun `gemm with zero inner dimension yields a zero matrix`() {
        val a = F64DenseMatrix(2, 0)
        val b = F64DenseMatrix(0, 3)
        assertEquals(F64DenseMatrix(2, 3), a * b)
    }

    @Test
    fun `invert produces the inverse of a general matrix`() {
        val rng = Random(20260805)
        for (n in intArrayOf(1, 2, 5, 9)) {
            val a = wellConditioned(n, rng)
            val inv = a.lu().invert()
            for (i in 0 until n) {
                for (j in 0 until n) {
                    var s = 0.0
                    for (k in 0 until n) s += a[i, k] * inv[k, j]
                    assertEquals(if (i == j) 1.0 else 0.0, s, 1e-9, "n=$n A·Ainv at [$i,$j]")
                }
            }
        }
    }

    @Test
    fun `the dense routines reject mismatched shapes`() {
        assertFailsWith<IllegalArgumentException> { F64DenseMatrix(2, 3).lu() }
        assertFailsWith<IllegalArgumentException> { F64DenseMatrix(2, 3) * F64DenseMatrix(2, 2) }
        assertFailsWith<IllegalArgumentException> { koblas.gemv(F64DenseMatrix(2, 3), DoubleArray(2)) }
        assertFailsWith<IllegalArgumentException> {
            koblas.gemv(
                F64DenseMatrix(2, 3),
                DoubleArray(3),
                transpose = true,
            )
        }
        assertFailsWith<DimensionMismatch> { F64DenseMatrix.diagonal(3).lu().solve(DoubleArray(2)) }
    }
}
