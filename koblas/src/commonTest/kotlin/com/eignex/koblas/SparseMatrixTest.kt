package com.eignex.koblas

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.koblas
import com.eignex.koblas.times
import com.eignex.koblas.transpose
import com.eignex.koblas.withColumn
import kotlin.test.*

class SparseMatrixTest {
    @Test
    fun `triplet construction agrees with the scalar coordinate reference`() {
        for (rows in intArrayOf(1, 3, 8)) {
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
        for (copies in intArrayOf(1, 2)) {
            val indices = IntArray(4 * copies) { (it / copies * 3 + 1) % 8 }
            val columns = IntArray(indices.size) { it / copies % 3 }
            val values = DoubleArray(indices.size) { it + 0.25 }
            val expected = coordinateReference(8, 3, indices, columns, values)
            val first = SparseMatrix.ofTriplets(8, 3, indices, columns, values)
            val second = SparseMatrix.ofTriplets(8, 3, indices, columns, values)

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
    fun `ofColumns sums duplicate entries and sorts rows`() {
        val a = SparseMatrix.ofColumns(3, 1, listOf(listOf(2 to 1.0, 0 to 2.0, 2 to 3.0)))
        assertTrue(intArrayOf(0, 2).contentEquals(a.rowIndices))
        assertTrue(doubleArrayOf(2.0, 4.0).contentEquals(a.values)) // 1.0 + 3.0 summed at row 2
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
    fun `rows may descend across a column boundary`() {
        // Within a column they must ascend, but the indices restart at each boundary.
        val ok = SparseMatrix(2, 2, intArrayOf(0, 1, 2), intArrayOf(1, 0), doubleArrayOf(5.0, 7.0))
        assertEquals(5.0, ok[1, 0])
        assertEquals(7.0, ok[0, 1])
    }

    @Test
    fun `ofTriplets builds the same matrix as ofColumns`() {
        val viaColumns = SparseMatrix.ofColumns(
            rows = 2,
            cols = 3,
            columns = listOf(listOf(0 to 1.0), listOf(1 to 3.0), listOf(0 to 2.0)),
        )
        val viaTriplets = SparseMatrix.ofTriplets(
            rows = 2,
            cols = 3,
            rowIndices = intArrayOf(0, 1, 0),
            colIndices = intArrayOf(0, 1, 2),
            values = doubleArrayOf(1.0, 3.0, 2.0),
        )
        assertEquals(viaTriplets, viaColumns)
    }

    @Test
    fun `ofTriplets orders arbitrary input and sums duplicates`() {
        val a = SparseMatrix.ofTriplets(
            rows = 3,
            cols = 3,
            rowIndices = intArrayOf(2, 1, 0, 1, 2),
            colIndices = intArrayOf(2, 1, 0, 1, 0),
            values = doubleArrayOf(9.0, 2.0, 1.0, 3.0, 7.0),
        )
        assertEquals(1.0, a[0, 0])
        assertEquals(7.0, a[2, 0])
        assertEquals(5.0, a[1, 1], "the two (1,1) entries should sum")
        assertEquals(9.0, a[2, 2])
        assertEquals(0.0, a[0, 1])
        assertEquals(4, a.nnz, "the duplicate should be merged, not stored twice")
        for (j in 0 until a.cols) {
            for (k in a.colPointers[j] + 1 until a.colPointers[j + 1]) {
                assertTrue(a.rowIndices[k - 1] < a.rowIndices[k], "column $j is not ascending")
            }
        }
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
    fun `triplet factories reject unrepresentable array lengths as shape errors`() {
        assertFailsWith<DimensionMismatch> {
            SparseMatrix.ofTriplets(0, Int.MAX_VALUE, IntArray(0), IntArray(0), DoubleArray(0))
        }
        assertFailsWith<DimensionMismatch> {
            SparseMatrix.ofTriplets(Int.MAX_VALUE, 1, intArrayOf(0), intArrayOf(0), doubleArrayOf(1.0))
        }
    }

    @Test
    fun `ofTriplets handles an empty entry set and rejects out-of-range positions`() {
        val empty = SparseMatrix.ofTriplets(2, 2, IntArray(0), IntArray(0), DoubleArray(0))
        assertEquals(0, empty.nnz)
        assertEquals(0.0, empty[1, 1])
        assertEquals(2, empty.rows)

        assertFailsWith<IndexOutOfBoundsException> {
            SparseMatrix.ofTriplets(2, 2, intArrayOf(2), intArrayOf(0), doubleArrayOf(1.0))
        }
        assertFailsWith<IndexOutOfBoundsException> {
            SparseMatrix.ofTriplets(2, 2, intArrayOf(0), intArrayOf(2), doubleArrayOf(1.0))
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
    fun `the transpose round-trips and preserves stored zeros`() {
        val a = SparseMatrix.ofColumns(
            3,
            2,
            listOf(
                listOf(0 to 4.0, 1 to 0.0, 2 to -1.0),
                listOf(1 to 7.0),
            ),
        )
        val t = a.transpose()
        assertEquals(2, t.rows)
        assertEquals(3, t.cols)
        assertEquals(a.nnz, t.nnz, "an explicitly stored zero was dropped")
        for (i in 0 until a.rows) {
            for (j in 0 until a.cols) assertEquals(a[i, j], t[j, i], "transpose at [$i,$j]")
        }
        assertEquals(a, t.transpose(), "transpose twice is not the original")
    }

    @Test
    fun `the transpose handles degenerate shapes`() {
        val empty = SparseMatrix.ofColumns(0, 0, emptyList())
        assertEquals(empty, empty.transpose().transpose())
        val noEntries = SparseMatrix.ofColumns(3, 2, listOf(emptyList(), emptyList()))
        val t = noEntries.transpose()
        assertEquals(2, t.rows)
        assertEquals(3, t.cols)
        assertEquals(0, t.nnz)
    }

    @Test
    fun `CSC mat-vec multiplies a matrix and its transpose`() {
        val a = SparseMatrix.ofColumns(
            rows = 2,
            cols = 3,
            columns = listOf(
                listOf(0 to 1.0),
                listOf(1 to 3.0),
                listOf(0 to 2.0),
            ),
        )
        assertContentEquals(doubleArrayOf(7.0, 9.0), koblas.gemv(a, doubleArrayOf(1.0, 3.0, 3.0)))
        assertContentEquals(doubleArrayOf(1.0, 6.0, 2.0), koblas.gemv(a, doubleArrayOf(1.0, 2.0), transpose = true))
    }

    @Test
    fun `wrap adopts CSC arrays`() {
        val a = SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(1, 0), doubleArrayOf(5.0, 7.0))
        assertEquals(5.0, a[1, 0])
        assertEquals(7.0, a[0, 1])
    }

    @Test
    fun `withColumn replaces one column and leaves the rest`() {
        val a = SparseMatrix.ofColumns(
            3,
            2,
            listOf(
                listOf(0 to 4.0, 2 to -1.0),
                listOf(1 to 7.0),
            ),
        )
        val b = a.withColumn(0, SparseVector.of(3, intArrayOf(1), doubleArrayOf(5.0)))
        assertEquals(0.0, b[0, 0])
        assertEquals(5.0, b[1, 0])
        assertEquals(0.0, b[2, 0])
        assertEquals(7.0, b[1, 1])
        assertEquals(2, b.nnz)
    }

    @Test
    fun `withColumn rejects an entering column of the wrong length`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))
        assertFailsWith<DimensionMismatch> {
            a.withColumn(0, SparseVector.of(3, intArrayOf(0), doubleArrayOf(1.0)))
        }
    }
}
