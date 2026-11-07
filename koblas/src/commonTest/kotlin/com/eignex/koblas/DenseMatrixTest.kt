package com.eignex.koblas

import kotlinx.serialization.json.*
import kotlin.test.*

class DenseMatrixTest {
    @Test
    fun `column views share only the logical entries of padded matrix views`() {
        val values = DoubleArray(42) { it * 0.25 - 3.0 }
        val matrix = DenseMatrix.wrap(6, 7, values).view(1, 2, 3, 2)
        val expected = values.copyOf()
        for (j in 0 until matrix.cols) {
            val reference = DenseVector.wrap(expected, matrix.offset + j * matrix.leadingDimension, matrix.rows)
            KoblasEngine.scalar.scale(reference, -2.0)
        }

        matrix.forEachColumns { column -> koblas.scale(column, -2.0) }

        assertColumnsAgreeWithReference(expected, values)
        val column = matrix.columnView(1)
        assertSame(values, column.values)
        assertEquals(matrix.rows, column.size)
        column[0] = 19.0
        assertEquals(19.0, matrix[0, 1])
        matrix[2, 1] = -7.0
        assertEquals(-7.0, column[2])
    }

    @Test
    fun `column traversal includes empty columns and supports empty matrix boundaries`() {
        val matrix = DenseMatrix.wrap(0, 3, DoubleArray(0), leadingDimension = Int.MAX_VALUE)
        val sizes = mutableListOf<Int>()

        matrix.forEachColumns { sizes += it.size }

        assertEquals(listOf(0, 0, 0), sizes)
        assertEquals(0, matrix.columnView(2).size)
        assertEquals(0, DenseMatrix.zero(0, Int.MAX_VALUE).columnView(Int.MAX_VALUE - 1).size)
        DenseMatrix.zero(3, 0).forEachColumns { error("Unexpected column") }
        assertFailsWith<IndexOutOfBoundsException> { matrix.columnView(-1) }
        assertFailsWith<IndexOutOfBoundsException> { matrix.columnView(3) }
    }

    private fun assertColumnsAgreeWithReference(expected: DoubleArray, actual: DoubleArray) {
        assertClose(expected, actual, "dense column writes")
    }


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

    @Test
    fun `wrap permits unused buffer entries without copying`() {
        val values = DoubleArray(6) { it.toDouble() }

        val matrix = DenseMatrix.wrap(2, 2, values)

        assertSame(values, matrix.values)
        assertEquals(0, matrix.offset)
        assertEquals(2, matrix.leadingDimension)
        assertContentEquals(doubleArrayOf(1.0, 3.0), matrix.toArray()[1])
    }

    @Test
    fun `equality and hashing ignore offsets and padding`() {
        val compact = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, -0.0), doubleArrayOf(Double.NaN, 3.0)))
        val parent = DenseMatrix.wrap(5, 4, DoubleArray(20) { Double.NaN })
        val view = parent.view(1, 1, 2, 2)
        view.copyFrom(compact)

        assertEquals(compact, view)
        assertEquals(view, compact)
        assertEquals(compact.hashCode(), view.hashCode())
        parent[0, 0] = 42.0
        assertEquals(compact, view)
        view[0, 1] = 0.0
        assertNotEquals(compact, view)
    }

    @Test
    fun `serialization writes only the visible rectangle and decodes compact storage`() {
        val parent = DenseMatrix.wrap(5, 4, DoubleArray(20) { Double.NaN })
        val view = parent.view(1, 1, 2, 3)
        for (j in 0 until view.cols) for (i in 0 until view.rows) view[i, j] = (i + 2 * j).toDouble()

        val encoded = Json.encodeToString(DenseMatrix.serializer(), view)
        val decoded = Json.decodeFromString(DenseMatrix.serializer(), encoded)
        val payload = Json.parseToJsonElement(encoded).jsonObject

        assertEquals(setOf("rows", "cols", "values"), payload.keys)
        assertEquals(List(6) { it.toDouble() }, payload.getValue("values").jsonArray.map { it.jsonPrimitive.double })
        assertEquals(view, decoded)
        assertEquals(0, decoded.offset)
        assertEquals(2, decoded.leadingDimension)
        assertEquals(6, decoded.values.size)
        assertNotSame(parent.values, decoded.values)
        decoded[0, 0] = 42.0
        assertEquals(0.0, view[0, 0])
    }

    @Test
    fun `views round trip through polymorphic matrix storage`() {
        val source: MatrixStorage = DenseMatrix.diagonal(4).view(1, 1, 2, 2)

        val encoded = Json.encodeToString(MatrixStorage.serializer(), source)
        val decoded = Json.decodeFromString(MatrixStorage.serializer(), encoded)

        assertEquals("DenseMatrix", Json.parseToJsonElement(encoded).jsonObject.getValue("type").jsonPrimitive.content)
        assertEquals(source, decoded)
        assertEquals(4, assertIs<DenseMatrix>(decoded).values.size)
    }

    @Test
    fun `empty equality hashing and serialization do not scan unused dimensions`() {
        val source = DenseMatrix.wrap(0, Int.MAX_VALUE, doubleArrayOf(Double.NaN), 1, Int.MAX_VALUE)
        val compact = DenseMatrix.zero(0, Int.MAX_VALUE)

        assertEquals(compact, source)
        assertEquals(compact.hashCode(), source.hashCode())
        val encoded = Json.encodeToString(DenseMatrix.serializer(), source)
        val decoded = Json.decodeFromString(DenseMatrix.serializer(), encoded)

        assertEquals(source, decoded)
        assertTrue(decoded.values.isEmpty())
    }

    @Test
    fun `decoding requires exactly the logical entry count`() {
        for (payload in listOf(
            """{"rows":2,"cols":2,"values":[1.0,2.0,3.0]}""",
            """{"rows":1,"cols":1,"values":[1.0,2.0]}""",
            """{"rows":-1,"cols":0,"values":[]}""",
        )) {
            assertFailsWith<IllegalArgumentException> { Json.decodeFromString(DenseMatrix.serializer(), payload) }
        }
    }

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
        val view = DenseMatrix.wrap(2, 3, values, offset = 2, leadingDimension = 5)

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
        DenseMatrix.wrap(0, Int.MAX_VALUE, doubleArrayOf(), leadingDimension = Int.MAX_VALUE)
        DenseMatrix.wrap(2, 0, doubleArrayOf(), leadingDimension = 2)
    }

    @Test
    fun `invalid raw layouts are rejected without integer overflow`() {
        val values = DoubleArray(10)
        for (create in listOf<() -> DenseMatrix>(
            { DenseMatrix.wrap(-1, 2, values) },
            { DenseMatrix.wrap(2, -1, values) },
            { DenseMatrix.wrap(2, 2, values, -1) },
            { DenseMatrix.wrap(0, 0, values, 11) },
            { DenseMatrix.wrap(2, 2, values, leadingDimension = 1) },
            { DenseMatrix.wrap(2, 3, values, 1, 4) },
            { DenseMatrix.wrap(2, Int.MAX_VALUE, values, 1, Int.MAX_VALUE) },
        )) {
            assertFailsWith<DimensionMismatch> { create() }
        }
    }

    @Test
    fun `nested rectangles and logical indices are bounded by the view`() {
        val view = DenseMatrix.zero(8, 8).view(2, 2, 3, 3)

        for (create in listOf<() -> DenseMatrix>(
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
}
