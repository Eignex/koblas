package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.matMul
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ScalingTest {

    private fun example() = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(1.0, 2.0, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
        ),
    )

    @Test
    fun `scaleRows is the product of the diagonal with the matrix`() {
        val d = doubleArrayOf(2.0, -1.0)
        val actual = example()
        actual.scaleRows(d)
        val expected = F64DenseMatrix.diagonal(d).matMul(example())
        assertEquals(expected, actual)
    }

    @Test
    fun `scaleColumns is the product of the matrix with the diagonal`() {
        val d = doubleArrayOf(2.0, -1.0, 0.5)
        val actual = example()
        actual.scaleColumns(d)
        val expected = example().matMul(F64DenseMatrix.diagonal(d))
        assertEquals(expected, actual)
    }

    @Test
    fun `the two directions compose to a two-sided scaling`() {
        val rowScale = doubleArrayOf(2.0, 3.0)
        val colScale = doubleArrayOf(1.0, -1.0, 0.5)
        val actual = example()
        actual.scaleRows(rowScale)
        actual.scaleColumns(colScale)
        val expected = F64DenseMatrix.diagonal(rowScale).matMul(example()).matMul(F64DenseMatrix.diagonal(colScale))
        assertEquals(expected, actual)
    }

    @Test
    fun `scaleColumns on CSC agrees with the dense result and keeps the pattern`() {
        val s = F64SparseMatrix.ofTriplets(
            rows = 2,
            cols = 3,
            rowIdx = intArrayOf(0, 1, 0),
            colIdx = intArrayOf(0, 1, 2),
            values = doubleArrayOf(1.0, 5.0, 3.0),
        )
        val d = doubleArrayOf(2.0, -1.0, 0.5)
        val nnzBefore = s.nnz
        s.scaleColumns(d)

        val dense = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(1.0, 0.0, 3.0), doubleArrayOf(0.0, 5.0, 0.0)),
        )
        dense.scaleColumns(d)
        for (i in 0 until 2) {
            // A stored zero scaled by a negative gives -0.0 while the sparse side has no entry and reads +0.0.
            for (j in 0 until 3) assertEquals(dense[i, j], s[i, j], 0.0, "($i,$j)")
        }
        assertEquals(nnzBefore, s.nnz, "a scaling must not change the pattern")
    }

    @Test
    fun `a unit diagonal leaves the matrix alone`() {
        val a = example()
        a.scaleRows(DoubleArray(2) { 1.0 })
        a.scaleColumns(DoubleArray(3) { 1.0 })
        assertEquals(example(), a)
    }

    @Test
    fun `a diagonal of the wrong length is rejected`() {
        assertFailsWith<DimensionMismatch> { example().scaleRows(DoubleArray(3)) }
        assertFailsWith<DimensionMismatch> { example().scaleColumns(DoubleArray(2)) }
        assertFailsWith<DimensionMismatch> {
            F64SparseMatrix.ofTriplets(2, 2, IntArray(0), IntArray(0), DoubleArray(0)).scaleColumns(DoubleArray(3))
        }
    }
}
