package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * The vector carriers: [DenseVector], [SparseVector] and the [VectorView] surface they share.
 *
 * The distinction that matters here is copy versus alias. A factory copies its input so a later mutation
 * of the caller's array cannot reach into the vector, while `wrap` deliberately relinquishes ownership and
 * aliases it; both directions are checked, because getting either wrong produces action at a distance.
 * Equality is by content across both carriers, so it is checked including the negative cases.
 */
class VectorTest {

    @Test
    fun `DenseVector zero factory builds a zero-filled vector`() {
        val z = DenseVector.zero(3)
        assertEquals(3, z.size)
        for (i in 0 until 3) assertEquals(0.0, z[i])
    }

    @Test
    fun `DenseVector of copies its input`() {
        val src = doubleArrayOf(1.0, 2.0, 3.0)
        val v = DenseVector.of(src)
        src[0] = 99.0
        assertEquals(1.0, v[0])
    }

    @Test
    fun `DenseVector wrap aliases the backing array without copying`() {
        val data = doubleArrayOf(1.0, 2.0)
        val v = DenseVector.wrap(data)
        data[1] = 42.0 // mutating the source is visible through the view (ownership relinquished)
        assertEquals(42.0, v[1], 0.0)
    }

    @Test
    fun `DenseVector toDoubleArray returns a copy`() {
        val v = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val copy = v.toDoubleArray()
        copy[0] = 99.0
        assertEquals(1.0, v[0])
    }

    @Test
    fun `DenseVector equals respects content`() {
        val a = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val c = DenseVector.of(doubleArrayOf(1.0, 3.0))
        assertEquals(a, a)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, c)
        assertNotEquals<Any?>(a, "not-a-vector")
    }

    @Test
    fun `DenseVector toString includes size`() {
        assertTrue("size=2" in DenseVector.zero(2).toString())
    }

    @Test
    fun `SparseVector rejects mismatched arrays`() {
        assertFailsWith<IllegalArgumentException> {
            SparseVector.of(5, intArrayOf(0, 1), doubleArrayOf(1.0))
        }
    }

    @Test
    fun `SparseVector of copies its inputs`() {
        val idx = intArrayOf(0, 2)
        val vals = doubleArrayOf(1.0, 3.0)
        val v = SparseVector.of(5, idx, vals)
        idx[0] = 4
        vals[1] = 99.0
        assertEquals(1.0, v[0])
        assertEquals(3.0, v[2])
    }

    @Test
    fun `SparseVector get returns zero for missing indices`() {
        val v = SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(5.0, 7.0))
        assertEquals(5.0, v[0])
        assertEquals(0.0, v[1])
        assertEquals(7.0, v[2])
        assertEquals(0.0, v[3])
    }

    @Test
    fun `SparseVector toDoubleArray materialises stored entries`() {
        val v = SparseVector.of(4, intArrayOf(1, 3), doubleArrayOf(2.0, 4.0))
        assertTrue(v.toDoubleArray().contentEquals(doubleArrayOf(0.0, 2.0, 0.0, 4.0)))
    }

    @Test
    fun `SparseVector equals and hashCode respect content`() {
        val a = SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0))
        val b = SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0))
        val different = SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(1.0, 4.0))
        val sizeDiff = SparseVector.of(5, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0))
        assertEquals(a, a)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, different)
        assertNotEquals(a, sizeDiff)
        assertNotEquals<Any?>(a, "x")
    }

    @Test
    fun `SparseVector toString includes nnz`() {
        val s = SparseVector.of(4, intArrayOf(0, 3), doubleArrayOf(1.0, 1.0)).toString()
        assertTrue("nnz=2" in s)
    }
}
