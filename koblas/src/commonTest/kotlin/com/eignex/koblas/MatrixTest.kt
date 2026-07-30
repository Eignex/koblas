package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * The [DenseMatrix] carrier and the [MatrixView] surface: factories, entry access, equality and shape
 * reporting.
 *
 * The storage is a single flat row-major array, so the factories are where a shape and its backing can
 * drift apart — a ragged input, an empty row set, or a `wrap` that aliases the caller's array. `toArray`
 * materializes a fresh copy in the other direction and is checked for that independence.
 */
class MatrixTest {

    @Test
    fun `DenseMatrix toArray on 0x0 returns empty array`() {
        val m = DenseMatrix(0, 0)
        assertEquals(0, m.toArray().size)
    }

    @Test
    fun `DenseMatrix of rejects ragged rows`() {
        assertFailsWith<IllegalArgumentException> {
            DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0)))
        }
    }

    @Test
    fun `DenseMatrix of supports an empty row set`() {
        val m = DenseMatrix.of(arrayOf())
        assertEquals(0, m.rows)
        assertEquals(0, m.cols)
    }

    @Test
    fun `DenseMatrix diagonal seeds entries`() {
        val m = DenseMatrix.diagonal(3, 2.5)
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(if (i == j) 2.5 else 0.0, m[i, j])
            }
        }
    }

    @Test
    fun `DenseMatrix wrap aliases the backing array without copying`() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val m = DenseMatrix.wrap(2, 2, data)
        data[0] = 99.0 // mutating the source is visible through the view (ownership relinquished)
        assertEquals(99.0, m[0, 0], 0.0)
    }

    @Test
    fun `DenseMatrix toArray returns a fresh copy`() {
        val m = DenseMatrix.diagonal(2, 1.0)
        val arr = m.toArray()
        arr[0][0] = 99.0
        assertEquals(1.0, m[0, 0])
    }

    @Test
    fun `DenseMatrix equals and hashCode respect content`() {
        val a = DenseMatrix.diagonal(2, 1.0)
        val b = DenseMatrix.diagonal(2, 1.0)
        val c = DenseMatrix.diagonal(2, 2.0)
        val sizeDiff = DenseMatrix.diagonal(3, 1.0)
        assertEquals(a, a)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, c)
        assertNotEquals(a, sizeDiff)
        assertNotEquals<Any?>(a, "x")
    }

    @Test
    fun `DenseMatrix toString shows shape`() {
        assertTrue("2x3" in DenseMatrix(2, 3).toString())
    }

    @Test
    fun `DenseMatrix rejects a shape its backing cannot hold`() {
        assertFailsWith<IllegalArgumentException> { DenseMatrix.wrap(2, 3, DoubleArray(5)) }
        assertFailsWith<IllegalArgumentException> { DenseMatrix(-1, 2) }
    }
}
