package com.eignex.koblas.core

import com.eignex.koblas.forEachStored
import kotlin.test.*

class VectorTest {

    @Test
    fun `F64DenseVector zero factory builds a zero-filled vector`() {
        val z = F64DenseVector.zero(3)
        assertEquals(3, z.size)
        for (i in 0 until 3) assertEquals(0.0, z[i])
    }

    @Test
    fun `F64DenseVector of copies its input`() {
        val src = doubleArrayOf(1.0, 2.0, 3.0)
        val v = F64DenseVector.of(src)
        src[0] = 99.0
        assertEquals(1.0, v[0])
    }

    @Test
    fun `F64DenseVector wrap aliases the backing array without copying`() {
        val data = doubleArrayOf(1.0, 2.0)
        val v = F64DenseVector.wrap(data)
        data[1] = 42.0 // mutating the source is visible through the wrapped vector
        assertEquals(42.0, v[1], 0.0)
    }

    @Test
    fun `F64DenseVector toDoubleArray returns a copy`() {
        val v = F64DenseVector.of(doubleArrayOf(1.0, 2.0))
        val copy = v.toDoubleArray()
        copy[0] = 99.0
        assertEquals(1.0, v[0])
    }

    @Test
    fun `F64DenseVector equals respects content`() {
        val a = F64DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = F64DenseVector.of(doubleArrayOf(1.0, 2.0))
        val c = F64DenseVector.of(doubleArrayOf(1.0, 3.0))
        assertEquals(a, a)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, c)
        assertNotEquals<Any?>(a, "not-a-vector")
    }

    @Test
    fun `F64SparseVector of copies its inputs`() {
        val idx = intArrayOf(0, 2)
        val vals = doubleArrayOf(1.0, 3.0)
        val v = F64SparseVector.of(5, idx, vals)
        idx[0] = 4
        vals[1] = 99.0
        assertEquals(1.0, v[0])
        assertEquals(3.0, v[2])
    }

    @Test
    fun `F64SparseVector get returns zero for missing indices`() {
        val v = F64SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(5.0, 7.0))
        assertEquals(5.0, v[0])
        assertEquals(0.0, v[1])
        assertEquals(7.0, v[2])
        assertEquals(0.0, v[3])
    }

    @Test
    fun `F64SparseVector toDoubleArray materialises stored entries`() {
        val v = F64SparseVector.of(4, intArrayOf(1, 3), doubleArrayOf(2.0, 4.0))
        assertTrue(v.toDoubleArray().contentEquals(doubleArrayOf(0.0, 2.0, 0.0, 4.0)))
    }

    @Test
    fun `F64SparseVector equals and hashCode respect content`() {
        val a = F64SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0))
        val b = F64SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0))
        val different = F64SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(1.0, 4.0))
        val sizeDiff = F64SparseVector.of(5, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0))
        assertEquals(a, a)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, different)
        assertNotEquals(a, sizeDiff)
        assertNotEquals<Any?>(a, "x")
    }

    @Test
    fun `of sorts by index and sums duplicates`() {
        val v = F64SparseVector.of(5, intArrayOf(3, 0, 3), doubleArrayOf(1.0, 2.0, 0.5))
        assertTrue(intArrayOf(0, 3).contentEquals(v.indices), "indices should be ascending: ${v.indices.toList()}")
        assertTrue(doubleArrayOf(2.0, 1.5).contentEquals(v.values), "duplicate at 3 should sum to 1.5")
        assertEquals(2.0, v[0])
        assertEquals(1.5, v[3])
        assertEquals(0.0, v[1])
        val seen = ArrayList<Int>()
        v.forEachStored { i, _ -> seen.add(i) }
        assertEquals(listOf(0, 3), seen)
    }

    @Test
    fun `get finds every stored entry by binary search`() {
        val idx = intArrayOf(0, 1, 4, 7, 8, 15)
        val v = F64SparseVector.of(16, idx, DoubleArray(idx.size) { it + 1.0 })
        for ((k, i) in idx.withIndex()) assertEquals(k + 1.0, v[i], "missed stored index $i")
        for (i in 0 until 16) if (i !in idx) assertEquals(0.0, v[i], "index $i should be absent")
    }

    @Test
    fun `copyIndices cannot mutate sparse structure`() {
        val v = F64SparseVector.of(4, intArrayOf(1, 3), doubleArrayOf(2.0, 4.0))
        val indices = v.copyIndices()

        indices[0] = 0

        assertEquals(2.0, v[1])
        assertEquals(0.0, v[0])
    }

    @Test
    fun `F64SparseVector wrap adopts ascending arrays and validates them`() {
        val indices = intArrayOf(0, 3)
        val values = doubleArrayOf(2.0, 1.5)
        val v = F64SparseVector.wrap(5, indices, values)
        assertEquals(2.0, v[0])
        assertEquals(1.5, v[3])
        assertEquals(0.0, v[1])
        assertSame(indices, v.indices)
        assertSame(values, v.values)
        assertFailsWith<IllegalArgumentException> {
            F64SparseVector.wrap(5, intArrayOf(3, 0), doubleArrayOf(1.0, 2.0))
        }
        assertFailsWith<IllegalArgumentException> {
            F64SparseVector.wrap(5, intArrayOf(0, 1), doubleArrayOf(1.0))
        }
    }
}
