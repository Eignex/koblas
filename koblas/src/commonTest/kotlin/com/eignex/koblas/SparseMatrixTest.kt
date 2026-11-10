package com.eignex.koblas

import kotlinx.serialization.json.Json
import kotlin.test.*

class SparseMatrixTest {
    @Test
    fun `column traversal scales stored values through the typed vector engine`() {
        val matrix = columnFixture()
        val expected = DenseVector.of(matrix.values)
        KoblasEngine.scalar.scale(expected, -2.0)
        val sizes = mutableListOf<Int>()

        matrix.forEachColumns { column ->
            sizes += column.size
            koblas.scale(column, -2.0)
        }

        assertEquals(listOf(4, 4, 4), sizes)
        assertColumnsAgreeWithReference(expected.values, matrix.values)
        assertContentEquals(intArrayOf(0, 2, 4, 5), matrix.copyColumnPointers())
    }

    @Test
    fun `column views follow insertions and replacements anywhere in their owner`() {
        val matrix = columnFixture()
        val first = matrix.columnView(0)
        val middle = matrix.columnView(1)
        val last = matrix.columnView(2)

        first[1] = 7.0
        middle[0] = 0.0
        last.copyFrom(SparseVector.wrap(4, intArrayOf(1, 3), doubleArrayOf(9.0, 0.0)))
        first.copyFrom(SparseVector.wrap(4, IntArray(0), DoubleArray(0)))

        assertContentEquals(intArrayOf(0, 0, 3, 5), matrix.copyColumnPointers())
        assertContentEquals(intArrayOf(0, 1, 2, 1, 3), matrix.copyRowIndices())
        assertContentEquals(doubleArrayOf(0.0, 3.0, 4.0, 9.0, 0.0), matrix.values)
        assertEquals(0, first.nnz)
        assertEquals(0, middle.offset)
        assertEquals(3, last.offset)
        assertSame(matrix.values, last.values)
        last[1] = -6.0
        assertEquals(-6.0, matrix[1, 2])
        matrix[2, 2] = 11.0
        assertEquals(11.0, last[2])
    }

    @Test
    fun `column axpy stages shared storage before changing the destination pattern`() {
        val matrix = columnFixture()
        val source = matrix.columnView(0)
        val destination = matrix.columnView(1)
        val last = matrix.columnView(2)
        val expected = DenseVector.wrap(destination.toDoubleArray())
        KoblasEngine.scalar.axpy(2.0, DenseVector.wrap(source.toDoubleArray()), expected)
        val firstBefore = source.toDoubleArray()
        val lastBefore = last.toDoubleArray()

        koblas.axpy(2.0, source, destination)

        assertColumnsAgreeWithReference(expected.values, destination.toDoubleArray())
        assertContentEquals(firstBefore, source.toDoubleArray())
        assertContentEquals(lastBefore, last.toDoubleArray())
        assertContentEquals(intArrayOf(0, 1, 2, 3), destination.copyIndices())
    }

    @Test
    fun `column traversal visits empty columns without scanning the row domain`() {
        val matrix = SparseMatrix.ofColumns(Int.MAX_VALUE, 3, listOf(emptyList(), listOf(5 to 2.0), emptyList()))
        val counts = mutableListOf<Int>()

        matrix.forEachColumns { column ->
            counts += column.nnz
            koblas.scale(column, 3.0)
        }

        assertEquals(listOf(0, 1, 0), counts)
        assertEquals(6.0, matrix[5, 1])
        SparseMatrix.ofColumns(3, 0, emptyList()).forEachColumns { error("Unexpected column") }
        assertFailsWith<IndexOutOfBoundsException> { matrix.columnView(-1) }
        assertFailsWith<IndexOutOfBoundsException> { matrix.columnView(3) }
    }

    private fun columnFixture(): SparseMatrix = SparseMatrix.wrap(
        4,
        3,
        intArrayOf(0, 2, 4, 5),
        intArrayOf(0, 3, 1, 2, 0),
        doubleArrayOf(2.0, -0.0, 3.0, 4.0, -5.0),
    )

    private fun assertColumnsAgreeWithReference(expected: DoubleArray, actual: DoubleArray) {
        assertClose(expected, actual, "sparse column writes")
    }

    @Test
    fun `writes through the matrix contract preserve validated CSC storage`() {
        val matrix = SparseMatrix.ofColumns(4, 3, listOf(listOf(2 to 2.0), emptyList(), listOf(0 to -1.0)))
        val mutable: Matrix = matrix

        mutable[3, 1] = 0.0
        mutable[1, 0] = 4.0
        val pointers = matrix.colPointers
        val indices = matrix.rowIndices
        val values = matrix.values
        mutable[0, 2] = 0.0

        assertContentEquals(intArrayOf(0, 2, 3, 4), matrix.colPointers)
        assertContentEquals(intArrayOf(1, 2, 3, 0), matrix.rowIndices)
        assertContentEquals(doubleArrayOf(4.0, 2.0, 0.0, 0.0), matrix.values)
        assertSame(pointers, matrix.colPointers)
        assertSame(indices, matrix.rowIndices)
        assertSame(values, matrix.values)
        val encoded = Json.encodeToString(SparseMatrix.serializer(), matrix)
        assertEquals(matrix, Json.decodeFromString(SparseMatrix.serializer(), encoded))
        assertFailsWith<IndexOutOfBoundsException> { mutable[-1, 0] = 1.0 }
        assertFailsWith<IndexOutOfBoundsException> { mutable[0, 3] = 1.0 }
    }

    @Test
    fun `triplet construction agrees with the scalar coordinate reference`() {
        for (rows in intArrayOf(1, 3, 8, 17, 18, 64, 65)) {
            for (columns in intArrayOf(1, 3, 6)) {
                for (count in intArrayOf(0, 1, 17, 64)) {
                    val indices = IntArray(count) { (it * 5 + 1) % rows }
                    val columnIndices = IntArray(count) { (it * 7 + 1) % columns }
                    val values = DoubleArray(count) { (it % 5 - 2).toDouble() }
                    val expected = coordinateReference(rows, columns, indices, columnIndices, values)

                    val actual = SparseMatrix.ofTriplets(rows, columns, indices, columnIndices, values)

                    assertTripletsAgreesWithReference(expected, actual)
                }
            }
        }
    }

    @Test
    fun `triplet construction preserves empty runs and stored zeros`() {
        val indices = intArrayOf(5, 1, 5, 1, 3, 1)
        val columns = intArrayOf(3, 1, 3, 3, 1, 1)
        val values = doubleArrayOf(1.5, -0.0, -1.5, 4.0, 0.0, -0.0)
        val expected = coordinateReference(8, 6, indices, columns, values)

        val actual = SparseMatrix.ofTriplets(8, 6, indices, columns, values)

        assertTripletsAgreesWithReference(expected, actual)
    }

    @Test
    fun `triplet results own arrays independently of inputs and other calls`() {
        for (rows in intArrayOf(8, Int.MAX_VALUE)) {
            for (copies in intArrayOf(1, 2)) {
                val indices = IntArray(4 * copies) { (it / copies * 3 + 1) % 8 }
                val columns = IntArray(indices.size) { it / copies % 3 }
                val values = DoubleArray(indices.size) { it + 0.25 }
                val reference = coordinateReference(8, 3, indices, columns, values)
                val expected = SparseMatrix.wrap(
                    rows,
                    3,
                    reference.copyColumnPointers(),
                    reference.copyRowIndices().map { if (it == 7) rows - 1 else it }.toIntArray(),
                    reference.values.copyOf(),
                )
                for (k in indices.indices) if (indices[k] == 7) indices[k] = rows - 1
                val first = SparseMatrix.ofTriplets(rows, 3, indices, columns, values)
                val second = SparseMatrix.ofTriplets(rows, 3, indices, columns, values)

                indices.fill(-1)
                columns.fill(-1)
                values.fill(-99.0)

                assertTripletsAgreesWithReference(expected, first)
                first.colPointers.fill(-1)
                first.rowIndices.fill(-1)
                first.values.fill(-99.0)
                assertTripletsAgreesWithReference(expected, second)
            }
        }
    }

    @Test
    fun `sparse factories preserve canonical support across row extents`() {
        val indices = intArrayOf(2, 0, 1, 2, 2, 0, 2, 1, 0, 0, 2, 2)
        val columns = intArrayOf(3, 1, 1, 1, 3, 3, 3, 3, 1, 3, 1, 4)
        val values = doubleArrayOf(1e16, -0.0, 0.0, 5.0, 1.0, 4.0, -1e16, -2.0, -0.0, -4.0, -5.0, 9.0)
        val reference = coordinateReference(3, 6, indices, columns, values)
        for (rows in intArrayOf(3, values.size, values.size + 1, Int.MAX_VALUE)) {
            val expandedRows = intArrayOf(0, rows / 2, rows - 1)
            val expected = SparseMatrix.wrap(
                rows,
                6,
                reference.copyColumnPointers(),
                reference.copyRowIndices().map { expandedRows[it] }.toIntArray(),
                reference.values.copyOf(),
            )
            val inputRows = indices.map { expandedRows[it] }.toIntArray()
            val columnData = List(6) { j ->
                indices.indices.filter { columns[it] == j }.map { inputRows[it] to values[it] }
            }

            val triplets = SparseMatrix.ofTriplets(rows, 6, inputRows, columns, values)
            val fromColumns = SparseMatrix.ofColumns(rows, 6, columnData)

            assertTripletsAgreesWithReference(expected, triplets)
            assertEquals(expected, fromColumns, "column construction with $rows rows")
        }
    }

    @Test
    fun `column results own storage independently of lists and other calls`() {
        for (rows in intArrayOf(3, Int.MAX_VALUE)) {
            val column = mutableListOf(rows - 1 to 2.0, 0 to 1.0, rows - 1 to -2.0)
            val columns = mutableListOf<List<Pair<Int, Double>>>(column, emptyList())
            val expected = SparseMatrix.wrap(
                rows,
                2,
                intArrayOf(0, 2, 2),
                intArrayOf(0, rows - 1),
                doubleArrayOf(1.0, 0.0),
            )
            val first = SparseMatrix.ofColumns(rows, 2, columns)
            val second = SparseMatrix.ofColumns(rows, 2, columns)

            column.clear()
            columns.clear()

            assertEquals(expected, first)
            first.colPointers.fill(-1)
            first.rowIndices.fill(-1)
            first.values.fill(-99.0)
            assertEquals(expected, second)
        }
    }

    private fun coordinateReference(
        rows: Int,
        columns: Int,
        indices: IntArray,
        columnIndices: IntArray,
        values: DoubleArray,
    ): SparseMatrix {
        val stored = BooleanArray(rows * columns)
        val dense = DoubleArray(stored.size)
        for (k in indices.indices) {
            val position = indices[k] + columnIndices[k] * rows
            dense[position] = if (stored[position]) dense[position] + values[k] else values[k]
            stored[position] = true
        }
        val pointers = IntArray(columns + 1)
        val resultRows = IntArray(stored.count { it })
        val resultValues = DoubleArray(resultRows.size)
        var count = 0
        for (j in 0 until columns) {
            for (i in 0 until rows) {
                if (stored[i + j * rows]) {
                    resultRows[count] = i
                    resultValues[count++] = dense[i + j * rows]
                }
            }
            pointers[j + 1] = count
        }
        return SparseMatrix.wrap(rows, columns, pointers, resultRows, resultValues)
    }

    private fun assertTripletsAgreesWithReference(expected: SparseMatrix, actual: SparseMatrix) {
        assertEquals(expected, actual, "triplet construction")
    }

    @Test
    fun `get reads stored entries and returns zero elsewhere`() {
        val a = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0)))
        assertEquals(1.0, a[0, 0])
        assertEquals(3.0, a[2, 0])
        assertEquals(2.0, a[1, 1])
        assertEquals(0.0, a[1, 0], "unstored entry must read as zero")
        assertEquals(0.0, a[0, 1])
        assertEquals(0.0, a[2, 1])
        assertFailsWith<IndexOutOfBoundsException> { a[3, 0] }
        assertFailsWith<IndexOutOfBoundsException> { a[0, 2] }
        assertFailsWith<IndexOutOfBoundsException> { a[-1, 0] }
        assertFailsWith<IndexOutOfBoundsException> { a[0, -1] }
    }

    @Test
    fun `stored column traversal rejects an index outside the shape`() {
        val a = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 1.0)))

        assertFailsWith<IndexOutOfBoundsException> { a.forEachInColumn(-1) { _, _ -> } }
        assertFailsWith<IndexOutOfBoundsException> { a.forEachInColumn(1) { _, _ -> } }
    }

    @Test
    fun `structural copies cannot mutate CSC storage`() {
        val a = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0)))
        val pointers = a.copyColumnPointers()
        val rows = a.copyRowIndices()

        pointers[1] = 0
        rows[0] = 1

        assertEquals(1.0, a[0, 0])
        assertEquals(3.0, a[2, 0])
        assertEquals(0.0, a[1, 0])
    }

    @Test
    fun `get finds a stored zero and equality distinguishes it from an absent one`() {
        val stored = SparseMatrix(2, 1, intArrayOf(0, 1), intArrayOf(1), doubleArrayOf(0.0))
        val absent = SparseMatrix(2, 1, intArrayOf(0, 0), IntArray(0), DoubleArray(0))
        assertEquals(0.0, stored[1, 0])
        assertEquals(0.0, absent[1, 0])
        assertEquals(1, stored.nnz)
        assertEquals(0, absent.nnz)
        assertTrue(stored != absent, "a stored zero differs from no entry at all")
    }

    @Test
    fun `toArray densifies to the logical matrix`() {
        val a = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0)))
        val rows = a.toArray()
        assertEquals(3, rows.size)
        assertTrue(doubleArrayOf(1.0, 0.0).contentEquals(rows[0]))
        assertTrue(doubleArrayOf(0.0, 2.0).contentEquals(rows[1]))
        assertTrue(doubleArrayOf(3.0, 0.0).contentEquals(rows[2]))
    }

    @Test
    fun `equality and hashCode are structural`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 2.0)))
        val same = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 2.0)))
        val different = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 2.5)))
        assertEquals(a, same)
        assertEquals(a.hashCode(), same.hashCode())
        assertTrue(a != different)
    }

    @Test
    fun `empty factories preserve tall shapes without row scratch`() {
        for (rows in intArrayOf(0, 2, Int.MAX_VALUE)) {
            for (cols in intArrayOf(0, 2)) {
                val expected = SparseMatrix.wrap(rows, cols, IntArray(cols + 1), IntArray(0), DoubleArray(0))

                val triplets = SparseMatrix.ofTriplets(rows, cols, IntArray(0), IntArray(0), DoubleArray(0))
                val columns = SparseMatrix.ofColumns(rows, cols, List(cols) { emptyList() })

                assertEquals(expected, triplets, "triplets ${rows}x$cols")
                assertEquals(expected, columns, "columns ${rows}x$cols")
            }
        }
    }

    @Test
    fun `triplet factories reject unrepresentable pointer lengths as shape errors`() {
        assertFailsWith<DimensionMismatch> {
            SparseMatrix.ofTriplets(0, Int.MAX_VALUE, IntArray(0), IntArray(0), DoubleArray(0))
        }
    }

    @Test
    fun `factories reject out of range positions and inconsistent shapes`() {
        for (rows in intArrayOf(2, Int.MAX_VALUE)) {
            for (invalidRow in intArrayOf(-1, rows)) {
                assertFailsWith<IndexOutOfBoundsException> {
                    SparseMatrix.ofTriplets(rows, 2, intArrayOf(invalidRow), intArrayOf(0), doubleArrayOf(1.0))
                }
                assertFailsWith<IndexOutOfBoundsException> {
                    SparseMatrix.ofColumns(rows, 2, listOf(listOf(invalidRow to 1.0), emptyList()))
                }
            }
            for (invalidColumn in intArrayOf(-1, 2)) {
                assertFailsWith<IndexOutOfBoundsException> {
                    SparseMatrix.ofTriplets(rows, 2, intArrayOf(0), intArrayOf(invalidColumn), doubleArrayOf(1.0))
                }
            }
        }
        assertFailsWith<DimensionMismatch> {
            SparseMatrix.ofTriplets(2, 2, intArrayOf(0, 1), intArrayOf(0), doubleArrayOf(1.0))
        }
        assertFailsWith<DimensionMismatch> { SparseMatrix.ofColumns(2, 2, emptyList()) }
        assertFailsWith<DimensionMismatch> {
            SparseMatrix.ofTriplets(-1, 1, IntArray(0), IntArray(0), DoubleArray(0))
        }
    }

    /** External CSC input must validate its pattern; trusted producers may rely on their own invariants. */
    @Test
    fun `wrap still rejects a pattern it cannot vouch for`() {
        assertFailsWith<DimensionMismatch>("a short column pointer array") {
            SparseMatrix.wrap(2, 2, intArrayOf(0, 1), IntArray(0), DoubleArray(0))
        }
        assertFailsWith<DimensionMismatch>("misaligned values") {
            SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 1), intArrayOf(0), DoubleArray(0))
        }
        assertFailsWith<IllegalArgumentException>("a nonzero column pointer head") {
            SparseMatrix.wrap(2, 2, intArrayOf(1, 1, 1), intArrayOf(0), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException>("a column pointer tail different from nnz") {
            SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 1), intArrayOf(0), doubleArrayOf(1.0, 2.0))
        }
        assertFailsWith<IllegalArgumentException>("descending column pointers") {
            SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 1), intArrayOf(0), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException>("rows that descend") {
            SparseMatrix.wrap(3, 1, intArrayOf(0, 2), intArrayOf(2, 0), doubleArrayOf(1.0, 2.0))
        }
        assertFailsWith<IllegalArgumentException>("a row stored twice") {
            SparseMatrix.wrap(3, 1, intArrayOf(0, 2), intArrayOf(1, 1), doubleArrayOf(1.0, 2.0))
        }
        assertFailsWith<IndexOutOfBoundsException>("a row outside the matrix") {
            SparseMatrix.wrap(2, 1, intArrayOf(0, 1), intArrayOf(5), doubleArrayOf(1.0))
        }
        assertFailsWith<DimensionMismatch>("negative rows") {
            SparseMatrix.wrap(-1, 1, intArrayOf(0, 0), IntArray(0), DoubleArray(0))
        }
        assertFailsWith<DimensionMismatch>("negative columns") {
            SparseMatrix.wrap(1, -1, intArrayOf(0), IntArray(0), DoubleArray(0))
        }
    }

    @Test
    fun `wrap adopts CSC arrays`() {
        val a = SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(1, 0), doubleArrayOf(5.0, 7.0))
        assertEquals(5.0, a[1, 0])
        assertEquals(7.0, a[0, 1])
    }

    @Test
    fun `round trips through its CSC arrays`() {
        for (a in listOf(
            SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0))),
            SparseMatrix(2, 1, intArrayOf(0, 1), intArrayOf(1), doubleArrayOf(0.0)), // a stored zero
            SparseMatrix(0, 0, intArrayOf(0), IntArray(0), DoubleArray(0)),
            SparseMatrix(4, 3, intArrayOf(0, 0, 0, 0), IntArray(0), DoubleArray(0)), // all-empty columns
        )) {
            val back = Json.decodeFromString(
                SparseMatrix.serializer(),
                Json.encodeToString(SparseMatrix.serializer(), a),
            )
            assertEquals(a, back, "round-trip changed $a")
        }
    }

    @Test
    fun `encodes its structure rather than a dense grid`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 5.0), emptyList()))
        val encoded = Json.encodeToString(SparseMatrix.serializer(), a)
        assertTrue(encoded.contains("colPointers"), "expected the CSC arrays in $encoded")
        assertTrue(encoded.contains("rowIndices"), "expected the CSC arrays in $encoded")
        assertTrue("0.0" !in encoded, "a structural zero leaked into the payload: $encoded")
    }
}
