package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import kotlin.test.*

class SparseMatrixOpsTest {

    private fun example(): F64SparseMatrix = F64SparseMatrix.ofTriplets(
        rows = 3,
        cols = 4,
        rowIdx = intArrayOf(0, 2, 1, 0, 2),
        colIdx = intArrayOf(0, 0, 1, 3, 3),
        values = doubleArrayOf(1.0, -2.0, 3.0, 0.0, -4.0),
    )

    private fun dense(a: F64SparseMatrix): F64DenseMatrix = F64DenseMatrix.of(a.toArray())

    @Test
    fun `sparse matrix operations agree with dense reference`() {
        val sparse = example()
        val dense = dense(sparse)

        assertEquals(dense.norm1(), sparse.norm1())
        assertEquals(dense.normInf(), sparse.normInf())
        assertEquals(dense.normFro(), sparse.normFro())
        for (j in 0 until sparse.cols) assertContentEquals(dense.column(j).data, sparse.column(j).toDoubleArray())
        for (i in 0 until sparse.rows) assertContentEquals(dense.row(i).data, sparse.row(i).toDoubleArray())

        val factors = doubleArrayOf(-2.0, 0.5, 3.0)
        sparse.scaleRows(factors)
        dense.scaleRows(factors)
        for (i in 0 until sparse.rows) {
            // Dense scaling changes an implicit +0.0 to -0.0 for a negative factor; CSC has no stored entry.
            for (j in 0 until sparse.cols) assertEquals(dense[i, j], sparse[i, j], 0.0, "($i,$j)")
        }
    }

    @Test
    fun `row scaling keeps CSC storage including explicit zeros`() {
        val sparse = example()
        val pointers = sparse.copyColumnPointers()
        val indices = sparse.copyRowIndices()

        sparse.scaleRows(doubleArrayOf(-2.0, 0.5, 3.0))

        assertContentEquals(pointers, sparse.copyColumnPointers())
        assertContentEquals(indices, sparse.copyRowIndices())
        assertEquals((-0.0).toBits(), sparse.values[3].toBits())
    }

    @Test
    fun `sparse row and column preserve explicit zeros and are independent`() {
        val sparse = example()
        val column = sparse.column(3)
        val row = sparse.row(0)

        assertIs<F64SparseVector>(column)
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
    fun `sparse norms handle empty shapes and NaN`() {
        for ((rows, cols) in listOf(0 to 0, 0 to 3, 4 to 0)) {
            val empty = F64SparseMatrix.ofTriplets(rows, cols, IntArray(0), IntArray(0), DoubleArray(0))
            assertEquals(0.0, empty.norm1())
            assertEquals(0.0, empty.normInf())
            assertEquals(0.0, empty.normFro())
        }

        val poisoned = F64SparseMatrix.ofTriplets(
            2,
            2,
            intArrayOf(0, 1),
            intArrayOf(0, 1),
            doubleArrayOf(Double.NaN, 4.0),
        )
        assertTrue(poisoned.norm1().isNaN())
        assertTrue(poisoned.normInf().isNaN())
        assertTrue(poisoned.normFro().isNaN())
        assertEquals(5e200, F64SparseMatrix.ofTriplets(1, 2, intArrayOf(0, 0), intArrayOf(0, 1), doubleArrayOf(3e200, 4e200)).normFro(), 1e188)
    }

    @Test
    fun `sparse extraction and row scaling validate shapes like dense`() {
        val sparse = example()
        assertFailsWith<IndexOutOfBoundsException> { sparse.column(-1) }
        assertFailsWith<IndexOutOfBoundsException> { sparse.column(4) }
        assertFailsWith<IndexOutOfBoundsException> { sparse.row(-1) }
        assertFailsWith<IndexOutOfBoundsException> { sparse.row(3) }
        assertFailsWith<DimensionMismatch> { sparse.scaleRows(DoubleArray(2)) }
    }

    @Test
    fun `sparse infinity norm reuses workspace`() {
        val sparse = example()
        val workspace = Workspace().apply { reserve(sparse.rows, count = 1) }
        assertEquals(sparse.normInf(), sparse.normInf(workspace))
        assertEquals(sparse.normInf(), sparse.normInf(workspace))
    }
}
