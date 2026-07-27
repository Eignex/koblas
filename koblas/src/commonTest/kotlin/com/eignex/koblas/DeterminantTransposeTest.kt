package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class DeterminantTransposeTest {

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
            val a = DenseMatrix(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
            for (i in 0 until n) a[i, i] = a[i, i] + n
            val cols = List(n) { j -> (0 until n).map { i -> i to a[i, j] } }
            val sparseDet = assertNotNull(SparseLu.factorize(SparseMatrix.ofColumns(n, n, cols))).determinant()
            val denseDet = a.lu().determinant()
            assertTrue(
                abs(denseDet - sparseDet) <= 1e-9 * maxOf(1.0, abs(denseDet)),
                "n=$n: dense $denseDet vs sparse $sparseDet",
            )
        }
    }

    @Test
    fun `transpose round-trips and maps entries`() {
        val a = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
            ),
        )
        val t = a.transpose()
        assertEquals(3, t.rows)
        assertEquals(2, t.cols)
        for (i in 0 until a.rows) for (j in 0 until a.cols) assertEquals(a[i, j], t[j, i])
        assertEquals(a, t.transpose())
        // Degenerate shapes.
        assertEquals(DenseMatrix(0, 0), DenseMatrix(0, 0).transpose())
        assertEquals(0, DenseMatrix(0, 5).transpose().cols)
        assertEquals(5, DenseMatrix(0, 5).transpose().rows)
    }

    @Test
    fun `transpose agrees with the gemm transpose flag`() {
        val rng = Random(20260804)
        val a = DenseMatrix(4, 6, DoubleArray(24) { rng.nextDouble(-1.0, 1.0) })
        val b = DenseMatrix(4, 3, DoubleArray(12) { rng.nextDouble(-1.0, 1.0) })
        // Aᵀ·B via materialized transpose vs the flag.
        val viaMaterialized = a.transpose().matMul(b)
        val viaFlag = DenseMatrix(6, 3)
        koblas.gemm(1.0, a, true, b, false, 0.0, viaFlag)
        for (idx in viaFlag.data.indices) {
            assertTrue(abs(viaMaterialized.data[idx] - viaFlag.data[idx]) < 1e-12)
        }
    }
}
