package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class MatrixProductsTest {
    @Test
    fun `generic dense product returns dense storage`() {
        val left: Matrix = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val right: Matrix = DenseMatrix.ofRows(arrayOf(doubleArrayOf(5.0), doubleArrayOf(6.0)))

        val result = left * right

        assertEquals(DenseMatrix::class, result::class)
        assertContentEquals(doubleArrayOf(17.0, 39.0), (result as DenseMatrix).values)
    }

    @Test
    fun `generic product supports transpose beta alias and reusable workspace`() {
        val backing = doubleArrayOf(1.0, 3.0, 2.0, 4.0)
        val left: Matrix = DenseMatrix.wrap(2, 2, backing)
        val right: Matrix = DenseMatrix.diagonal(2, 2.0)
        val destination = DenseMatrix.wrap(2, 2, backing)

        left.gemmInto(1.0, true, right, false, 0.5, destination, MatrixWorkspace())

        assertContentEquals(doubleArrayOf(2.5, 5.5, 7.0, 10.0), destination.values)
    }

    @Test
    fun `sparse route fails before destination mutation until S2`() {
        val sparse: Matrix = SparseMatrix.ofColumns(2, 2, listOf(emptyList(), emptyList()))
        val dense: Matrix = DenseMatrix.diagonal(2)
        val destination = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val before = destination.values.copyOf()

        assertFailsWith<UnsupportedOperationException> {
            sparse.gemmInto(1.0, false, dense, false, 0.0, destination)
        }

        assertContentEquals(before, destination.values)
    }
}
