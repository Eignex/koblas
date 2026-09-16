package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class StridedViewsTest {
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
        assertFailsWith<DimensionMismatch> { StridedVector(DoubleArray(3), 2, 2) }
        assertFailsWith<IllegalArgumentException> { StridedVector(DoubleArray(3), 0, 2, 0) }
    }

    @Test
    fun `a borrowed slice reaches the same kernels as dense storage`() {
        // Every second entry of the backing array, so the run the kernels walk is not contiguous.
        val backing = DoubleArray(8) { it + 1.0 }
        val even = StridedVector(backing, 0, 4, 2)
        val odd = StridedVector(backing, 1, 4, 2)

        assertEquals(1.0 * 2 + 3.0 * 4 + 5.0 * 6 + 7.0 * 8, even dot odd)
        even.scale(2.0)

        assertContentEquals(doubleArrayOf(2.0, 2.0, 6.0, 4.0, 10.0, 6.0, 14.0, 8.0), backing)
    }
}
