package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LinearAlgebraEdgeTest {

    @Test
    fun `1x1 solve and transpose-solve`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(4.0)))
        assertEquals(0.5, a.lu().solve(doubleArrayOf(2.0))[0], 1e-12)
        assertEquals(0.5, a.lu().solve(doubleArrayOf(2.0), transpose = true)[0], 1e-12)
    }

    @Test
    fun `a 1x1 zero matrix is singular`() {
        assertTrue(DenseMatrix.of(arrayOf(doubleArrayOf(0.0))).lu().singular)
    }

    @Test
    fun `gemm with zero inner dimension yields a zero matrix`() {
        val a = DenseMatrix(2, 0)
        val b = DenseMatrix(0, 3)
        assertEquals(DenseMatrix(2, 3), a.matMul(b))
    }

    @Test
    fun `factor rejects a non-square matrix`() {
        assertFailsWith<IllegalArgumentException> { DenseMatrix(2, 3).lu() }
    }

    @Test
    fun `gemm rejects mismatched inner dimensions`() {
        assertFailsWith<IllegalArgumentException> { DenseMatrix(2, 3).matMul(DenseMatrix(2, 2)) }
    }

    @Test
    fun `gemv rejects a mismatched vector length`() {
        assertFailsWith<IllegalArgumentException> { koblas.gemv(DenseMatrix(2, 3), DoubleArray(2)) }
        assertFailsWith<IllegalArgumentException> { koblas.gemv(DenseMatrix(2, 3), DoubleArray(3), transpose = true) }
    }

    @Test
    fun `solve rejects a mismatched right-hand side length`() {
        assertFailsWith<IllegalArgumentException> { DenseMatrix.diagonal(3).lu().solve(DoubleArray(2)) }
    }

    @Test
    fun `wrap aliases the backing array without copying`() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val m = DenseMatrix.wrap(2, 2, data)
        data[0] = 99.0 // mutating the source is visible through the view (ownership relinquished)
        assertEquals(99.0, m[0, 0], 0.0)
        val v = DenseVector.wrap(data)
        data[1] = 42.0
        assertEquals(42.0, v[1], 0.0)
    }

    @Test
    fun `SparseMatrix rejects structurally invalid CSC`() {
        assertFailsWith<IllegalArgumentException> {
            SparseMatrix(
                2,
                2,
                intArrayOf(0, 1),
                intArrayOf(0),
                doubleArrayOf(1.0),
            )
        }
        // rowIdx out of range.
        assertFailsWith<IllegalArgumentException> {
            SparseMatrix(2, 1, intArrayOf(0, 1), intArrayOf(5), doubleArrayOf(1.0))
        }
    }

    @Test
    fun `ofColumns sums duplicate entries and sorts rows`() {
        val a = SparseMatrix.ofColumns(3, 1, listOf(listOf(2 to 1.0, 0 to 2.0, 2 to 3.0)))
        assertTrue(intArrayOf(0, 2).contentEquals(a.rowIdx)) // ascending
        assertTrue(doubleArrayOf(2.0, 4.0).contentEquals(a.values)) // 1.0 + 3.0 summed at row 2
    }
}
