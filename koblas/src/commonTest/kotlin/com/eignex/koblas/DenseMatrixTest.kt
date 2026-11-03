package com.eignex.koblas

import kotlinx.serialization.json.*
import kotlin.test.*

class DenseMatrixTest {

    @Test
    fun `DenseMatrix of rejects ragged rows`() {
        assertFailsWith<DimensionMismatch> {
            DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0)))
        }
    }

    @Test
    fun `DenseMatrix of supports an empty row set`() {
        val m = DenseMatrix.ofRows(arrayOf())
        assertEquals(0, m.rows)
        assertEquals(0, m.cols)
        assertEquals(0, m.toArray().size)
    }

    @Test
    fun `DenseMatrix wrap aliases the backing array without copying`() {
        val values = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val m = DenseMatrix.wrap(2, 2, values)
        values[0] = 99.0 // mutating the source is visible through the wrapped matrix
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
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, c)
        assertNotEquals(a, sizeDiff)
        assertNotEquals<Any?>(a, "x")
    }

    @Test
    fun `DenseMatrix rejects a shape its backing cannot hold`() {
        assertFailsWith<DimensionMismatch> { DenseMatrix.wrap(2, 3, DoubleArray(5)) }
        assertFailsWith<DimensionMismatch> { DenseMatrix.wrap(2, 2, DoubleArray(6)) }
        assertFailsWith<DimensionMismatch> { DenseMatrix(-1, 2) }
        assertFailsWith<DimensionMismatch> { DenseMatrix(2, -1) }
        assertFailsWith<DimensionMismatch> { DenseMatrix.zero(65536, 65536) }
        assertFailsWith<DimensionMismatch> { DenseMatrix.zero(46341, 46341) }
        assertFailsWith<DimensionMismatch> { DenseMatrix.zero(2, 1073741824) }
    }

    @Test
    fun `DenseMatrix indexing rejects every address outside its shape`() {
        val matrix = DenseMatrix.ofRows(
            arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)),
        )
        assertFailsWith<IndexOutOfBoundsException> { matrix[2, 0] }
        assertFailsWith<IndexOutOfBoundsException> { matrix[5, 0] }
        assertFailsWith<IndexOutOfBoundsException> { matrix[-1, 1] }
        assertFailsWith<IndexOutOfBoundsException> { matrix[0, 3] }
        assertFailsWith<IndexOutOfBoundsException> { matrix[0, -1] }
        assertFailsWith<IndexOutOfBoundsException> { matrix[5, 0] = 0.0 }
        assertEquals(6.0, matrix[1, 2])
    }

    @Test
    fun `DenseMatrix ofColumns reads the outer index as the column`() {
        val m = DenseMatrix.ofColumns(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)))
        assertEquals(3, m.rows)
        assertEquals(2, m.cols)
        assertEquals(1.0, m[0, 0])
        assertEquals(3.0, m[2, 0])
        assertEquals(4.0, m[0, 1])
        assertEquals(6.0, m[2, 1])
        assertTrue(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).contentEquals(m.values))
    }

    @Test
    fun `DenseMatrix ofColumns rejects ragged columns and accepts none`() {
        assertFailsWith<DimensionMismatch> {
            DenseMatrix.ofColumns(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0)))
        }
        val empty = DenseMatrix.ofColumns(arrayOf())
        assertEquals(0, empty.rows)
        assertEquals(0, empty.cols)
    }

    @Test
    fun `DenseMatrix zero allocates a zero matrix and defaults to square`() {
        val m = DenseMatrix.zero(2, 3)
        assertEquals(2, m.rows)
        assertEquals(3, m.cols)
        assertTrue(m.values.all { it == 0.0 })
        val square = DenseMatrix.zero(2)
        assertEquals(2, square.rows)
        assertEquals(2, square.cols)
    }

    @Test
    fun `DenseMatrix diagonal from a vector places each entry on the diagonal`() {
        val m = DenseMatrix.diagonal(doubleArrayOf(1.0, -2.0, 0.5))
        assertEquals(3, m.rows)
        assertEquals(3, m.cols)
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(if (i == j) doubleArrayOf(1.0, -2.0, 0.5)[i] else 0.0, m[i, j], "($i,$j)")
            }
        }
        assertEquals(DenseMatrix.diagonal(3, 2.5), DenseMatrix.diagonal(DoubleArray(3) { 2.5 }))
        assertEquals(0, DenseMatrix.diagonal(DoubleArray(0)).rows)
    }

    @Test
    fun `DenseMatrix round trips through JSON for square rectangular and degenerate shapes`() {
        val cases = listOf(
            DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))),
            DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0, 3.0))),
            DenseMatrix(1, 1).also { it[0, 0] = 7.5 },
            DenseMatrix(3, 0), // 3 empty rows survive (cols=0)
            DenseMatrix(0, 0),
        )
        for (m in cases) {
            val back = Json.decodeFromString(DenseMatrix.serializer(), Json.encodeToString(DenseMatrix.serializer(), m))
            assertEquals(m, back, "round-trip $m")
        }
    }

    @Test
    fun `DenseMatrix wire form is its shape and a flat array`() {
        val m = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val obj = Json.parseToJsonElement(Json.encodeToString(DenseMatrix.serializer(), m)).jsonObject
        assertEquals(2, obj.getValue("rows").jsonPrimitive.int)
        assertEquals(2, obj.getValue("cols").jsonPrimitive.int)
        assertEquals(
            listOf(1.0, 3.0, 2.0, 4.0),
            obj.getValue("values").jsonArray.map { it.jsonPrimitive.double },
        )
    }
}
