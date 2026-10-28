package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

class DenseMatrixViewTest {
    @Test
    fun `nested views share the parent buffer and spacing`() {
        val parent = DenseMatrix.wrap(7, 6, DoubleArray(42) { it.toDouble() })
        val view = parent.view(1, 2, 5, 4).view(2, 1, 2, 2)

        view[1, 1] = -10.0

        assertSame(parent.values, view.values)
        assertEquals(24, view.offset)
        assertEquals(7, view.leadingDimension)
        assertEquals(parent[3, 3], view[0, 0])
        assertEquals(-10.0, parent[4, 4])
        assertContentEquals(doubleArrayOf(parent[3, 3], parent[3, 4]), view.toArray()[0])
    }

    @Test
    fun `raw views accept padding and an exact last entry`() {
        val values = DoubleArray(14) { it.toDouble() }
        val view = DenseMatrixView.wrap(2, 3, values, offset = 2, leadingDimension = 5)

        assertEquals(13.0, view[1, 2])
        values[2] = -1.0
        assertEquals(-1.0, view[0, 0])
    }

    @Test
    fun `empty views retain their shape without requiring stored entries`() {
        val parent = DenseMatrix.zero(4, 3)
        val rowEmpty = parent.view(4, 0)
        val columnEmpty = parent.view(0, 3)
        val nested = columnEmpty.view(4, 0)

        assertEquals(0, rowEmpty.rows)
        assertEquals(3, rowEmpty.cols)
        assertEquals(4, columnEmpty.rows)
        assertEquals(0, columnEmpty.cols)
        assertEquals(0, nested.rows)
        assertEquals(0, nested.cols)
        DenseMatrixView.wrap(0, Int.MAX_VALUE, doubleArrayOf(), leadingDimension = Int.MAX_VALUE)
        DenseMatrixView.wrap(2, 0, doubleArrayOf(), leadingDimension = 2)
    }

    @Test
    fun `invalid raw layouts are rejected without integer overflow`() {
        val values = DoubleArray(10)
        for (create in listOf<() -> DenseMatrixView>(
            { DenseMatrixView.wrap(-1, 2, values) },
            { DenseMatrixView.wrap(2, -1, values) },
            { DenseMatrixView.wrap(2, 2, values, -1) },
            { DenseMatrixView.wrap(0, 0, values, 11) },
            { DenseMatrixView.wrap(2, 2, values, leadingDimension = 1) },
            { DenseMatrixView.wrap(2, 3, values, 1, 4) },
            { DenseMatrixView.wrap(2, Int.MAX_VALUE, values, 1, Int.MAX_VALUE) },
        )) {
            assertFailsWith<DimensionMismatch> { create() }
        }
    }

    @Test
    fun `nested rectangles and logical indices are bounded by the view`() {
        val view = DenseMatrix.zero(8, 8).view(2, 2, 3, 3)

        for (create in listOf<() -> DenseMatrixView>(
            { view.view(-1, 0) },
            { view.view(0, 4) },
            { view.view(2, 0, 2, 1) },
            { view.view(0, 0, Int.MAX_VALUE, 1) },
            { view.view(0, 0, -1, 1) },
        )) {
            assertFailsWith<DimensionMismatch> { create() }
        }
        assertFailsWith<IndexOutOfBoundsException> { view[3, 0] }
        assertFailsWith<IndexOutOfBoundsException> { view[0, -1] = 1.0 }
    }

    @Test
    fun `contiguous wrapping still requires an exact buffer`() {
        assertFailsWith<DimensionMismatch> { DenseMatrix.wrap(2, 2, DoubleArray(6)) }
    }
}
