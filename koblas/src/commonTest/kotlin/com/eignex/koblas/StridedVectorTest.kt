package com.eignex.koblas

import com.eignex.koblas.internal.storage.overlaps
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.fail

class StridedVectorTest {
    // Strides of equal and unequal magnitude in both directions, origins inside and at the end of the buffer,
    // and empty, singleton and longer views reach every branch of the overlap test.
    @Test
    fun `overlap agrees with physical index intersection`() {
        val backing = DoubleArray(8)
        val views = buildList {
            for (stride in intArrayOf(-2, -1, 1, 2, 3)) {
                for (offset in intArrayOf(0, 1, 4, backing.size)) {
                    for (size in intArrayOf(0, 1, 2, 4)) {
                        val last = offset + (size - 1) * stride
                        if (size == 0 || (offset in backing.indices && last in backing.indices)) {
                            add(StridedVector(backing, offset, size, stride))
                        }
                    }
                }
            }
        }
        val addressed = views.map { view -> (0 until view.size).map { view.offset + it * view.stride }.toSet() }

        for (i in views.indices) {
            for (j in views.indices) {
                val expected = addressed[i].any { it in addressed[j] }

                if (views[i].overlaps(views[j]) != expected) fail("${views[i]} against ${views[j]}: expected $expected")
            }
        }
    }

    @Test
    fun `overlap handles extreme singleton strides and independent buffers`() {
        val backing = DoubleArray(2)
        for (stride in intArrayOf(Int.MIN_VALUE, Int.MAX_VALUE)) {
            val singleton = StridedVector(backing, 0, 1, stride)

            assertEquals(true, singleton.overlaps(StridedVector(backing, 0, 2)))
            assertEquals(false, singleton.overlaps(StridedVector(backing, 1, 1, stride)))
            assertEquals(false, singleton.overlaps(StridedVector(backing, 0, 0, stride)))
            assertEquals(false, singleton.overlaps(StridedVector(backing.copyOf(), 0, 1, stride)))
        }
    }

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

    @Test
    fun `a negative stride presents a reversed borrowed slice`() {
        val vector = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))

        val reversed = vector.view(offset = 4, size = 3, stride = -2)
        reversed[1] = -3.0

        assertContentEquals(doubleArrayOf(5.0, -3.0, 1.0), reversed.toDoubleArray())
        assertContentEquals(doubleArrayOf(1.0, 2.0, -3.0, 4.0, 5.0), vector.values)
    }

    @Test
    fun `view construction rejects addresses outside the buffer`() {
        assertFailsWith<DimensionMismatch> { StridedVector(DoubleArray(3), 2, 2) }
        assertFailsWith<IllegalArgumentException> { StridedVector(DoubleArray(3), 0, 2, 0) }
    }

    // Building the result from the buffer alone would move the window silently: every entry read would still
    // be in range and no call would fail, the answer would just be about different numbers.
    @Test
    fun `a view of a strided vector composes with the window it already had`() {
        val backing = DoubleArray(12) { it.toDouble() }
        val outer: DenseVector = StridedVector(backing, offset = 2, size = 4, stride = 2)

        assertContentEquals(doubleArrayOf(2.0, 4.0, 6.0, 8.0), outer.view(0, outer.size).toDoubleArray())
        assertContentEquals(doubleArrayOf(4.0, 8.0), outer.view(offset = 1, size = 2, stride = 2).toDoubleArray())
    }
}
