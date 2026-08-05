package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.lapackFailedAt
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

    /**
     * `failedAt` reports *where* the factorization broke down, which the sparse side has always carried and
     * the dense side used to compute and discard. The position is the first zero pivot, matching what
     * `dgetrf` returns in `info`: an all-zero matrix has several, and picking the last one would be just as
     * easy to implement and wrong.
     */
    @Test
    fun `factor reports the first zero pivot position`() {
        assertEquals(NOT_SINGULAR, DenseMatrix.diagonal(3).lu().failedAt, "a good factorization has no position")
        // Rank 1: after the row swap the second pivot cancels to exactly zero.
        val rank1 = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        assertEquals(1, rank1.lu().failedAt)
        assertEquals(0, DenseMatrix.of(arrayOf(doubleArrayOf(0.0))).lu().failedAt)
        assertEquals(0, DenseMatrix(3, 3).lu().failedAt, "the first of three zero pivots, not the last")
    }

    /** The `info`-to-position translation every LAPACK-backed backend shares: `info` is 1-based, and
     *  non-positive means no singularity to report. */
    @Test
    fun `lapackFailedAt converts an info return`() {
        assertEquals(NOT_SINGULAR, lapackFailedAt(0))
        assertEquals(0, lapackFailedAt(1))
        assertEquals(4, lapackFailedAt(5))
        assertEquals(NOT_SINGULAR, lapackFailedAt(-3), "an illegal-argument report is not a singularity")
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
