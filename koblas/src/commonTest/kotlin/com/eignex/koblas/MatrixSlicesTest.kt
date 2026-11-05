package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals

class MatrixSlicesTest {
    @Test
    fun `dense transpose round trips and maps entries`() {
        val a = DenseMatrix.ofRows(
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
        assertEquals(DenseMatrix(0, 0), DenseMatrix(0, 0).transpose())
        assertEquals(0, DenseMatrix(0, 5).transpose().cols)
        assertEquals(5, DenseMatrix(0, 5).transpose().rows)
    }

    @Test
    fun `sparse transpose round trips and preserves stored zeros`() {
        val a = SparseMatrix.ofColumns(
            3,
            2,
            listOf(
                listOf(0 to 4.0, 1 to 0.0, 2 to -1.0),
                listOf(1 to 7.0),
            ),
        )
        val t = a.transpose()
        assertEquals(2, t.rows)
        assertEquals(3, t.cols)
        assertEquals(a.nnz, t.nnz, "an explicitly stored zero was dropped")
        for (i in 0 until a.rows) {
            for (j in 0 until a.cols) assertEquals(a[i, j], t[j, i], "transpose at [$i,$j]")
        }
        assertEquals(a, t.transpose(), "transpose twice is not the original")
    }

    @Test
    fun `sparse transpose handles degenerate shapes`() {
        val empty = SparseMatrix.ofColumns(0, 0, emptyList())
        assertEquals(empty, empty.transpose().transpose())
        val noEntries = SparseMatrix.ofColumns(3, 2, listOf(emptyList(), emptyList()))
        val t = noEntries.transpose()
        assertEquals(2, t.rows)
        assertEquals(3, t.cols)
        assertEquals(0, t.nnz)
    }
}
