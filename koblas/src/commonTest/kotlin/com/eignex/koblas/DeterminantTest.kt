package com.eignex.koblas

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The determinant read off an LU factorization, dense and sparse.
 *
 * A determinant is the product of the pivots times the sign of the row permutation, so the sign is the
 * part worth pinning down: it comes from counting swaps, and a miscount produces a correct magnitude
 * with the wrong sign. The hand-computed values below include a swapped pair for exactly that reason,
 * and the two factorizations are compared against each other because they count swaps differently.
 */
class DeterminantTest {

    @Test
    fun `dense LU determinant matches hand values and tracks row-swap sign`() {
        // [[2, 1], [1, 3]] → det 5.
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        assertEquals(5.0, a.lu().determinant(), 1e-12)
        // Swapped rows negate: [[1, 3], [2, 1]] → det -5.
        val b = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 1.0)))
        assertEquals(-5.0, b.lu().determinant(), 1e-12)
        // 3x3 hand value: det = 1·(50−48) − 2·(40−42) + 3·(32−35) = 2 + 4 − 9 = −3.
        val c = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
                doubleArrayOf(7.0, 8.0, 10.0),
            ),
        )
        assertEquals(-3.0, c.lu().determinant(), 1e-9)
        // Empty and identity edge cases.
        assertEquals(1.0, DenseMatrix(0, 0).lu().determinant())
        assertEquals(1.0, DenseMatrix.diagonal(4).lu().determinant(), 1e-12)
    }

    @Test
    fun `dense LU determinant is exactly zero for a singular matrix`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        val lu = a.lu()
        assertTrue(lu.singular)
        assertEquals(0.0, lu.determinant())
    }

    @Test
    fun `dense and sparse determinants agree on random matrices`() {
        val rng = Random(20260803)
        for (n in intArrayOf(1, 3, 7, 15)) {
            val a = wellConditioned(n, rng)
            val cols = List(n) { j -> (0 until n).map { i -> i to a[i, j] } }
            val sparseDet = assertNotNull(SparseLu.factorize(SparseMatrix.ofColumns(n, n, cols))).determinant()
            assertClose(a.lu().determinant(), sparseDet, "n=$n dense vs sparse determinant", tolerance = 1e-9)
        }
    }
}
