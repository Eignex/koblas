package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class MatrixTest {

    @Test
    fun `DenseMatrix toArray on 0x0 returns empty array`() {
        val m = DenseMatrix(0, 0)
        assertEquals(0, m.toArray().size)
    }

    @Test
    fun `DenseMatrix of rejects ragged rows`() {
        assertFailsWith<DimensionMismatch> {
            DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0)))
        }
    }

    @Test
    fun `DenseMatrix of supports an empty row set`() {
        val m = DenseMatrix.of(arrayOf())
        assertEquals(0, m.rows)
        assertEquals(0, m.cols)
    }

    @Test
    fun `DenseMatrix diagonal seeds entries`() {
        val m = DenseMatrix.diagonal(3, 2.5)
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(if (i == j) 2.5 else 0.0, m[i, j])
            }
        }
    }

    @Test
    fun `DenseMatrix wrap aliases the backing array without copying`() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val m = DenseMatrix.wrap(2, 2, data)
        data[0] = 99.0 // mutating the source is visible through the view (ownership relinquished)
        assertEquals(99.0, m[0, 0], 0.0)
    }

    @Test
    fun `DenseMatrix toArray returns a fresh copy`() {
        val m = DenseMatrix.diagonal(2, 1.0)
        val arr = m.toArray()
        arr[0][0] = 99.0
        assertEquals(1.0, m[0, 0])
    }

    @Test
    fun `DenseMatrix equals and hashCode respect content`() {
        val a = DenseMatrix.diagonal(2, 1.0)
        val b = DenseMatrix.diagonal(2, 1.0)
        val c = DenseMatrix.diagonal(2, 2.0)
        val sizeDiff = DenseMatrix.diagonal(3, 1.0)
        assertEquals(a, a)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, c)
        assertNotEquals(a, sizeDiff)
        assertNotEquals<Any?>(a, "x")
    }

    @Test
    fun `DenseMatrix rejects a shape its backing cannot hold`() {
        assertFailsWith<DimensionMismatch> { DenseMatrix.wrap(2, 3, DoubleArray(5)) }
        assertFailsWith<DimensionMismatch> { DenseMatrix(-1, 2) }
    }

    @Test
    fun `column and row read the two axes of the same matrix`() {
        val m = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)))
        assertTrue(doubleArrayOf(2.0, 5.0).contentEquals(m.column(1).data), "column 1")
        assertTrue(doubleArrayOf(4.0, 5.0, 6.0).contentEquals(m.row(1).data), "row 1")
        for (j in 0 until m.cols) {
            val col = m.column(j)
            for (i in 0 until m.rows) assertEquals(m[i, j], col[i], "column $j entry $i")
        }
        for (i in 0 until m.rows) {
            val row = m.row(i)
            for (j in 0 until m.cols) assertEquals(m[i, j], row[j], "row $i entry $j")
        }
    }

    @Test
    fun `column copies rather than aliasing the backing`() {
        val m = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val col = m.column(0)
        col[0] = 99.0
        assertEquals(1.0, m[0, 0], "writing the copy must not reach the matrix")
    }

    @Test
    fun `column and row reject an index outside the shape`() {
        val m = DenseMatrix.zero(2, 3)
        assertFailsWith<IndexOutOfBoundsException> { m.column(3) }
        assertFailsWith<IndexOutOfBoundsException> { m.column(-1) }
        assertFailsWith<IndexOutOfBoundsException> { m.row(2) }
        assertFailsWith<IndexOutOfBoundsException> { m.row(-1) }
    }

    @Test
    fun `DenseMatrix ofColumns reads the outer index as the column`() {
        val m = DenseMatrix.ofColumns(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        assertEquals(2, m.rows)
        assertEquals(2, m.cols)
        assertEquals(1.0, m[0, 0])
        assertEquals(2.0, m[1, 0])
        assertEquals(3.0, m[0, 1])
        assertEquals(4.0, m[1, 1])
        assertTrue(doubleArrayOf(1.0, 2.0, 3.0, 4.0).contentEquals(m.data))
    }

    @Test
    fun `DenseMatrix ofColumns is the transpose of of`() {
        val entries = arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0))
        val byRows = DenseMatrix.of(entries)
        val byCols = DenseMatrix.ofColumns(entries)
        assertEquals(byRows.rows, byCols.cols)
        assertEquals(byRows.cols, byCols.rows)
        for (i in 0 until byRows.rows) {
            for (j in 0 until byRows.cols) assertEquals(byRows[i, j], byCols[j, i], "($i,$j)")
        }
    }

    @Test
    fun `DenseMatrix ofColumns rejects ragged columns and accepts none`() {
        assertFailsWith<DimensionMismatch> {
            DenseMatrix.ofColumns(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0)))
        }
        val empty = DenseMatrix.ofColumns(arrayOf())
        assertEquals(0, empty.rows)
        assertEquals(0, empty.cols)
    }

    @Test
    fun `DenseMatrix zero allocates a zero matrix and defaults to square`() {
        val m = DenseMatrix.zero(2, 3)
        assertEquals(2, m.rows)
        assertEquals(3, m.cols)
        assertTrue(m.data.all { it == 0.0 })
        val square = DenseMatrix.zero(2)
        assertEquals(2, square.rows)
        assertEquals(2, square.cols)
    }

    @Test
    fun `DenseMatrix diagonal from a vector places each entry on the diagonal`() {
        val m = DenseMatrix.diagonal(doubleArrayOf(1.0, -2.0, 0.5))
        assertEquals(3, m.rows)
        assertEquals(3, m.cols)
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(if (i == j) doubleArrayOf(1.0, -2.0, 0.5)[i] else 0.0, m[i, j], "($i,$j)")
            }
        }
        assertEquals(DenseMatrix.diagonal(3, 2.5), DenseMatrix.diagonal(DoubleArray(3) { 2.5 }))
        assertEquals(0, DenseMatrix.diagonal(DoubleArray(0)).rows)
    }
}
