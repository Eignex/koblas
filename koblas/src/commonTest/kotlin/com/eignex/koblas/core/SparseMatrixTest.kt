package com.eignex.koblas.core

import com.eignex.koblas.times
import com.eignex.koblas.*
import com.eignex.koblas.core.*
import com.eignex.koblas.sparse.gemv
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SparseMatrixTest {

    @Test
    fun `CSC mat-vec multiplies a matrix and its transpose`() {
        val a = F64SparseMatrix.ofColumns(
            rows = 2,
            cols = 3,
            columns = listOf(
                listOf(0 to 1.0),
                listOf(1 to 3.0),
                listOf(0 to 2.0),
            ),
        )
        assertTrue(doubleArrayOf(7.0, 9.0).contentEquals(a.gemv(doubleArrayOf(1.0, 3.0, 3.0))))
        assertTrue(doubleArrayOf(1.0, 6.0, 2.0).contentEquals(a.gemv(doubleArrayOf(1.0, 2.0), transpose = true)))
    }

    @Test
    fun `ofColumns sums duplicate entries and sorts rows`() {
        val a = F64SparseMatrix.ofColumns(3, 1, listOf(listOf(2 to 1.0, 0 to 2.0, 2 to 3.0)))
        assertTrue(intArrayOf(0, 2).contentEquals(a.rowIdx)) // ascending
        assertTrue(doubleArrayOf(2.0, 4.0).contentEquals(a.values)) // 1.0 + 3.0 summed at row 2
    }

    @Test
    fun `get reads stored entries and returns zero elsewhere`() {
        val a = F64SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0)))
        assertEquals(1.0, a[0, 0])
        assertEquals(3.0, a[2, 0])
        assertEquals(2.0, a[1, 1])
        assertEquals(0.0, a[1, 0], "unstored entry must read as zero")
        assertEquals(0.0, a[0, 1])
        assertEquals(0.0, a[2, 1])
        assertFailsWith<IndexOutOfBoundsException> { a[3, 0] }
        assertFailsWith<IndexOutOfBoundsException> { a[0, 2] }
    }

    @Test
    fun `structural copies cannot mutate CSC storage`() {
        val a = F64SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0)))
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
        val stored = F64SparseMatrix(2, 1, intArrayOf(0, 1), intArrayOf(1), doubleArrayOf(0.0))
        val absent = F64SparseMatrix(2, 1, intArrayOf(0, 0), IntArray(0), DoubleArray(0))
        assertEquals(0.0, stored[1, 0])
        assertEquals(0.0, absent[1, 0])
        assertEquals(1, stored.nnz)
        assertEquals(0, absent.nnz)
        assertTrue(stored != absent, "a stored zero differs from no entry at all")
    }

    @Test
    fun `toArray densifies to the logical matrix`() {
        val a = F64SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0)))
        val rows = a.toArray()
        assertEquals(3, rows.size)
        assertTrue(doubleArrayOf(1.0, 0.0).contentEquals(rows[0]))
        assertTrue(doubleArrayOf(0.0, 2.0).contentEquals(rows[1]))
        assertTrue(doubleArrayOf(3.0, 0.0).contentEquals(rows[2]))
    }

    @Test
    fun `equality and hashCode are structural`() {
        val a = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 2.0)))
        val same = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 2.0)))
        val different = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 2.5)))
        assertEquals(a, same)
        assertEquals(a.hashCode(), same.hashCode())
        assertTrue(a != different)
    }

    @Test
    fun `the F64MatrixView gemv overload agrees with the dense equivalent`() {
        val a = F64SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0)))
        val dense = F64DenseMatrix.of(a.toArray())
        for (x in listOf<F64VectorView>(
            F64DenseVector.of(doubleArrayOf(2.0, -1.0)),
            F64SparseVector.of(2, intArrayOf(1), doubleArrayOf(-1.0)),
            F64SparseVector.of(2, IntArray(0), DoubleArray(0)),
        )) {
            val viaSparse = a * x
            val viaDense = dense * x
            assertEquals(viaDense, viaSparse, "gemv disagreed for $x")
        }
        assertTrue(
            a.gemv(doubleArrayOf(2.0, -1.0)).contentEquals((a * F64DenseVector.of(doubleArrayOf(2.0, -1.0))).data),
        )
    }

    @Test
    fun `rows may descend across a column boundary`() {
        // Within a column they must ascend, but the indices restart at each boundary.
        val ok = F64SparseMatrix(2, 2, intArrayOf(0, 1, 2), intArrayOf(1, 0), doubleArrayOf(5.0, 7.0))
        assertEquals(5.0, ok[1, 0])
        assertEquals(7.0, ok[0, 1])
    }

    @Test
    fun `ofTriplets builds the same matrix as ofColumns`() {
        val viaColumns = F64SparseMatrix.ofColumns(
            rows = 2,
            cols = 3,
            columns = listOf(listOf(0 to 1.0), listOf(1 to 3.0), listOf(0 to 2.0)),
        )
        val viaTriplets = F64SparseMatrix.ofTriplets(
            rows = 2,
            cols = 3,
            rowIdx = intArrayOf(0, 1, 0),
            colIdx = intArrayOf(0, 1, 2),
            values = doubleArrayOf(1.0, 3.0, 2.0),
        )
        assertEquals(viaColumns, viaTriplets)
    }

    @Test
    fun `ofTriplets orders arbitrary input and sums duplicates`() {
        val a = F64SparseMatrix.ofTriplets(
            rows = 3,
            cols = 3,
            rowIdx = intArrayOf(2, 1, 0, 1, 2),
            colIdx = intArrayOf(2, 1, 0, 1, 0),
            values = doubleArrayOf(9.0, 2.0, 1.0, 3.0, 7.0),
        )
        assertEquals(1.0, a[0, 0])
        assertEquals(7.0, a[2, 0])
        assertEquals(5.0, a[1, 1], "the two (1,1) entries should sum")
        assertEquals(9.0, a[2, 2])
        assertEquals(0.0, a[0, 1])
        assertEquals(4, a.nnz, "the duplicate should be merged, not stored twice")
        for (j in 0 until a.cols) {
            for (k in a.colPtr[j] + 1 until a.colPtr[j + 1]) {
                assertTrue(a.rowIdx[k - 1] < a.rowIdx[k], "column $j is not ascending")
            }
        }
    }

    @Test
    fun `ofTriplets handles an empty entry set and rejects out-of-range positions`() {
        val empty = F64SparseMatrix.ofTriplets(2, 2, IntArray(0), IntArray(0), DoubleArray(0))
        assertEquals(0, empty.nnz)
        assertEquals(0.0, empty[1, 1])
        assertEquals(2, empty.rows)

        assertFailsWith<IllegalArgumentException> {
            F64SparseMatrix.ofTriplets(2, 2, intArrayOf(2), intArrayOf(0), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException> {
            F64SparseMatrix.ofTriplets(2, 2, intArrayOf(0), intArrayOf(2), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException> {
            F64SparseMatrix.ofTriplets(2, 2, intArrayOf(0, 1), intArrayOf(0), doubleArrayOf(1.0))
        }
    }

    @Test
    fun `the transpose round-trips and preserves stored zeros`() {
        val a = F64SparseMatrix.ofColumns(
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
        val empty = F64SparseMatrix.ofColumns(0, 0, emptyList())
        assertEquals(empty, empty.transpose().transpose())
        val noEntries = F64SparseMatrix.ofColumns(3, 2, listOf(emptyList(), emptyList()))
        val t = noEntries.transpose()
        assertEquals(2, t.rows)
        assertEquals(3, t.cols)
        assertEquals(0, t.nnz)
    }

    @Test
    fun `wrap adopts CSC arrays`() {
        val a = F64SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(1, 0), doubleArrayOf(5.0, 7.0))
        assertEquals(5.0, a[1, 0])
        assertEquals(7.0, a[0, 1])
    }
}
