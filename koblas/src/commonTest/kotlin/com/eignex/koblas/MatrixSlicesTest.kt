package com.eignex.koblas

import kotlin.test.*

class MatrixSlicesTest {
    @Test
    fun `sparse column agrees with dense reference`() {
        val sparse = sparseStorageExample()
        val dense = sparse.denseCopy()
        for (j in 0 until sparse.cols) assertContentEquals(dense.column(j).values, sparse.column(j).toDoubleArray())
    }

    @Test
    fun `sparse row agrees with dense reference`() {
        val sparse = sparseStorageExample()
        val dense = sparse.denseCopy()
        for (i in 0 until sparse.rows) assertContentEquals(dense.row(i).values, sparse.row(i).toDoubleArray())
    }

    @Test
    fun `sparse row and column preserve explicit zeros and are independent`() {
        val sparse = sparseStorageExample()
        val column = sparse.column(3)
        val row = sparse.row(0)

        assertIs<SparseVector>(column)
        assertContentEquals(intArrayOf(0, 2), column.copyIndices())
        assertContentEquals(doubleArrayOf(0.0, -4.0), column.values)
        assertContentEquals(intArrayOf(0, 3), row.copyIndices())
        assertContentEquals(doubleArrayOf(1.0, 0.0), row.values)

        column.values[0] = 9.0
        row.values[0] = 8.0
        assertEquals(0.0, sparse[0, 3])
        assertEquals(1.0, sparse[0, 0])
    }

    @Test
    fun `row and column return no stored entries for a row or column with none`() {
        val sparse = SparseMatrix.ofTriplets(
            rows = 4,
            cols = 3,
            rowIndices = intArrayOf(0, 3),
            colIndices = intArrayOf(0, 2),
            values = doubleArrayOf(1.0, 2.0),
        )

        val emptyRow = sparse.row(1)
        assertContentEquals(IntArray(0), emptyRow.copyIndices())
        assertContentEquals(DoubleArray(3), emptyRow.toDoubleArray())

        val emptyColumn = sparse.column(1)
        assertContentEquals(IntArray(0), emptyColumn.copyIndices())
        assertContentEquals(DoubleArray(4), emptyColumn.toDoubleArray())
    }

    @Test
    fun `row and column handle empty shapes`() {
        val noRows = SparseMatrix.ofTriplets(0, 3, IntArray(0), IntArray(0), DoubleArray(0))
        for (j in 0 until noRows.cols) assertContentEquals(DoubleArray(0), noRows.column(j).toDoubleArray())

        val noCols = SparseMatrix.ofTriplets(4, 0, IntArray(0), IntArray(0), DoubleArray(0))
        for (i in 0 until noCols.rows) assertContentEquals(DoubleArray(0), noCols.row(i).toDoubleArray())
    }

    @Test
    fun `sparse extraction validates row and column indices`() {
        val sparse = sparseStorageExample()
        assertFailsWith<IndexOutOfBoundsException> { sparse.column(-1) }
        assertFailsWith<IndexOutOfBoundsException> { sparse.column(4) }
        assertFailsWith<IndexOutOfBoundsException> { sparse.row(-1) }
        assertFailsWith<IndexOutOfBoundsException> { sparse.row(3) }
    }

    @Test
    fun `column and row read the two axes of the same matrix`() {
        val m = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)))
        assertTrue(doubleArrayOf(2.0, 5.0).contentEquals(m.column(1).values), "column 1")
        assertTrue(doubleArrayOf(4.0, 5.0, 6.0).contentEquals(m.row(1).values), "row 1")
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
        val m = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
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

    /** CSC fixture with empty columns, irregular rows, and an explicitly stored zero. */
    private fun sparseStorageExample(): SparseMatrix = SparseMatrix.ofTriplets(
        rows = 3,
        cols = 4,
        rowIndices = intArrayOf(0, 2, 1, 0, 2),
        colIndices = intArrayOf(0, 0, 1, 3, 3),
        values = doubleArrayOf(1.0, -2.0, 3.0, 0.0, -4.0),
    )
}
