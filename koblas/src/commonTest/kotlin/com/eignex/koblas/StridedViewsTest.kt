package com.eignex.koblas

import kotlin.test.*

class StridedViewsTest {
    @Test
    fun `a panel borrows its parent buffer and leading dimension`() {
        val matrix = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(0.0, 10.0, 20.0),
                doubleArrayOf(1.0, 11.0, 21.0),
                doubleArrayOf(2.0, 12.0, 22.0),
                doubleArrayOf(3.0, 13.0, 23.0),
            ),
        )

        val panel = matrix.view(row = 1, rows = 2, column = 1, cols = 2)
        panel[1, 1] = -1.0

        assertSame(matrix.data, panel.data)
        assertEquals(4, panel.leadingDimension)
        assertTrue(
            arrayOf(doubleArrayOf(11.0, 21.0), doubleArrayOf(12.0, -1.0)).contentDeepEquals(panel.toArray()),
        )
        assertEquals(-1.0, matrix[2, 2])
    }

    @Test
    fun `rows columns and overlapping panels remain live`() {
        val matrix = DenseMatrix.zero(4, 3)
        val left = matrix.view(0, 4, 0, 2)
        val overlap = matrix.view(1, 2, 1, 2)

        left.column(1)[2] = 7.0
        assertEquals(7.0, overlap[1, 0])

        overlap.row(0)[1] = 9.0
        assertEquals(9.0, matrix[1, 2])
    }

    @Test
    fun `a negative stride presents a reversed borrowed slice`() {
        val vector = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))

        val reversed = vector.view(offset = 4, size = 3, stride = -2)
        reversed[1] = -3.0

        assertContentEquals(doubleArrayOf(5.0, -3.0, 1.0), reversed.toDoubleArray())
        assertEquals(-3.0, vector[2])
    }

    @Test
    fun `view construction rejects addresses outside the buffer`() {
        assertFailsWith<IllegalArgumentException> { StridedVectorView(DoubleArray(3), 2, 2) }
        assertFailsWith<IllegalArgumentException> { StridedVectorView(DoubleArray(3), 0, 2, 0) }
        assertFailsWith<IllegalArgumentException> { StridedMatrixView(2, 2, DoubleArray(4), leadingDimension = 3) }
        assertFailsWith<IllegalArgumentException> { DenseMatrix.zero(2).view(1, 2, 0, 1) }
    }

    @Test
    fun `empty boundary panels keep a valid buffer offset`() {
        val backing = DoubleArray(8)
        val parents = listOf(
            DenseMatrix.zero(2, 3).asView(),
            StridedMatrixView(2, 2, backing, offset = 2, leadingDimension = 4),
            StridedMatrixView(0, Int.MAX_VALUE, DoubleArray(0), leadingDimension = Int.MAX_VALUE),
        )

        for (parent in parents) {
            val panels = listOf(
                parent.view(parent.rows, 0, parent.cols, 0),
                parent.view(0, parent.rows, parent.cols, 0),
                parent.view(parent.rows, 0, 0, parent.cols),
            )

            for (panel in panels) {
                assertSame(parent.data, panel.data)
                assertEquals(parent.leadingDimension, panel.leadingDimension)
                assertTrue(panel.offset in 0..panel.data.size)
                assertFalse(panel.overlaps(parent))
            }
        }
    }

    @Test
    fun `rows and columns of zero extent matrices produce empty vectors`() {
        val noRows = StridedMatrixView(0, 3, DoubleArray(0))
        val noColumns = StridedMatrixView(3, 0, DoubleArray(0))

        for (j in 0 until noRows.cols) {
            val column = noRows.column(j)
            assertSame(noRows.data, column.data)
            assertContentEquals(DoubleArray(0), column.toDoubleArray())
        }
        for (i in 0 until noColumns.rows) {
            val row = noColumns.row(i)
            assertSame(noColumns.data, row.data)
            assertContentEquals(DoubleArray(0), row.toDoubleArray())
        }
    }
}
