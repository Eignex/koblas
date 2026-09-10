package com.eignex.koblas

import kotlin.test.*

class MatrixSlicesTest {
    @Test
    fun `sparse column agrees with dense reference`() {
        val sparse = sparseStorageExample()
        val dense = sparse.denseCopy()
        for (j in 0 until sparse.cols) assertContentEquals(dense.column(j).data, sparse.column(j).toDoubleArray())
    }

    @Test
    fun `sparse row agrees with dense reference`() {
        val sparse = sparseStorageExample()
        val dense = sparse.denseCopy()
        for (i in 0 until sparse.rows) assertContentEquals(dense.row(i).data, sparse.row(i).toDoubleArray())
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
            rowIdx = intArrayOf(0, 3),
            colIdx = intArrayOf(0, 2),
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
}
