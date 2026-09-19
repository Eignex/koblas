package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class StridedVectorTest {
    @Test
    fun `nested views cannot escape the logical parent`() {
        val parent = StridedVector(DoubleArray(12), 4, 3)

        for ((offset, size, stride) in listOf(Triple(-1, 1, 1), Triple(3, 1, 1), Triple(1, 3, 1), Triple(1, 3, -1))) {
            assertFailsWith<DimensionMismatch> { parent.view(offset, size, stride) }
        }
    }

    @Test
    fun `nested view offset arithmetic cannot wrap into the buffer`() {
        val parent = StridedVector(DoubleArray(3), 0, 2, 2)

        assertFailsWith<DimensionMismatch> { parent.view(Int.MIN_VALUE, 1) }
    }

    @Test
    fun `nested view stride arithmetic cannot wrap into another stride`() {
        val parent = StridedVector(DoubleArray(1), 0, 1, Int.MAX_VALUE)

        assertFailsWith<DimensionMismatch> { parent.view(0, 1, 3) }
    }

    @Test
    fun `empty nested views accept both logical endpoints`() {
        for (stride in intArrayOf(2, -2)) {
            val parent = StridedVector(DoubleArray(5), if (stride < 0) 4 else 0, 3, stride)
            for (offset in intArrayOf(0, parent.size)) {
                val empty = parent.view(offset, 0)

                assertEquals(0, empty.size)
                assertContentEquals(doubleArrayOf(), empty.toDoubleArray())
            }
            assertFailsWith<DimensionMismatch> { parent.view(-1, 0) }
            assertFailsWith<DimensionMismatch> { parent.view(parent.size + 1, 0) }
        }
    }

    @Test
    fun `nested views reject negative sizes and zero strides`() {
        val parent = DenseVector.zero(3)

        assertFailsWith<DimensionMismatch> { parent.view(0, -1) }
        assertFailsWith<IllegalArgumentException> { parent.view(0, 1, 0) }
    }
}
