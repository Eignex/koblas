package com.eignex.koblas.core

import kotlin.test.*

class F64StridedViewsTest {
    @Test
    fun `a panel borrows its parent buffer and leading dimension`() {
        val matrix = F64DenseMatrix.of(
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
        val matrix = F64DenseMatrix.zero(4, 3)
        val left = matrix.view(0, 4, 0, 2)
        val overlap = matrix.view(1, 2, 1, 2)

        left.column(1)[2] = 7.0
        assertEquals(7.0, overlap[1, 0])

        overlap.row(0)[1] = 9.0
        assertEquals(9.0, matrix[1, 2])
    }

    @Test
    fun `a negative stride presents a reversed borrowed slice`() {
        val vector = F64DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))

        val reversed = vector.view(offset = 4, size = 3, stride = -2)
        reversed[1] = -3.0

        assertContentEquals(doubleArrayOf(5.0, -3.0, 1.0), reversed.toDoubleArray())
        assertEquals(-3.0, vector[2])
    }

    @Test
    fun `view construction rejects addresses outside the buffer`() {
        assertFailsWith<IllegalArgumentException> { F64StridedVectorView(DoubleArray(3), 2, 2) }
        assertFailsWith<IllegalArgumentException> { F64StridedVectorView(DoubleArray(3), 0, 2, 0) }
        assertFailsWith<IllegalArgumentException> { F64StridedMatrixView(2, 2, DoubleArray(4), leadingDimension = 3) }
        assertFailsWith<IllegalArgumentException> { F64DenseMatrix.zero(2).view(1, 2, 0, 1) }
    }
}
