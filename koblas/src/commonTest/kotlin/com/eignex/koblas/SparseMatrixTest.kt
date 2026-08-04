package com.eignex.koblas

import com.eignex.koblas.sparse.gemv
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The CSC matrix itself: its product with a vector, and the structural invariants of the format.
 *
 * Compressed-sparse-column storage is three parallel arrays whose consistency is not checkable by the
 * type system, so the constructor validates it and that validation is tested directly. The `ofColumns`
 * factory is the forgiving entry point: it accepts entries in any order and sums duplicates, which is
 * what callers assembling a matrix column by column need.
 */
class SparseMatrixTest {

    @Test
    fun `CSC mat-vec multiplies a matrix and its transpose`() {
        // [[1, 0, 2], [0, 3, 0]]  (2x3), CSC by column.
        val a = SparseMatrix.ofColumns(
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
    fun `SparseMatrix rejects structurally invalid CSC`() {
        // colPtr must have cols + 1 entries.
        assertFailsWith<IllegalArgumentException> {
            SparseMatrix(2, 2, intArrayOf(0, 1), intArrayOf(0), doubleArrayOf(1.0))
        }
        // rowIdx out of range.
        assertFailsWith<IllegalArgumentException> {
            SparseMatrix(2, 1, intArrayOf(0, 1), intArrayOf(5), doubleArrayOf(1.0))
        }
    }

    @Test
    fun `ofColumns sums duplicate entries and sorts rows`() {
        val a = SparseMatrix.ofColumns(3, 1, listOf(listOf(2 to 1.0, 0 to 2.0, 2 to 3.0)))
        assertTrue(intArrayOf(0, 2).contentEquals(a.rowIdx)) // ascending
        assertTrue(doubleArrayOf(2.0, 4.0).contentEquals(a.values)) // 1.0 + 3.0 summed at row 2
    }

    @Test
    fun `get reads stored entries and returns zero elsewhere`() {
        // column 0: rows 0 and 2; column 1: row 1 only.
        val a = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0)))
        assertEquals(1.0, a[0, 0])
        assertEquals(3.0, a[2, 0])
        assertEquals(2.0, a[1, 1])
        assertEquals(0.0, a[1, 0], "unstored entry must read as zero")
        assertEquals(0.0, a[0, 1])
        assertEquals(0.0, a[2, 1])
        assertFailsWith<IllegalArgumentException> { a[3, 0] }
        assertFailsWith<IllegalArgumentException> { a[0, 2] }
    }

    /** An explicitly stored zero is part of the value, as it is for SparseVector: the pattern is data. */
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

    /**
     * A sparse operand gives the same answer as its dense twin through the view overload, over both
     * storages of `x` including an all-empty one.
     *
     * What this does *not* check is that the sparse branch is taken. Deleting it would leave the generic
     * fallback, which reads through [MatrixView.get] and costs `rows × cols` column searches instead of
     * `nnz` work — and produces bit-identical results, so no assertion here would notice. That property
     * is a performance contract carried by the comment at the call site, not by this test; the only test
     * that could catch it would be one slow enough to hang rather than fail.
     */
    @Test
    fun `the MatrixView gemv overload agrees with the dense equivalent`() {
        val a = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0)))
        val dense = DenseMatrix.of(a.toArray())
        for (x in listOf<VectorView>(
            DenseVector.of(doubleArrayOf(2.0, -1.0)),
            SparseVector.of(2, intArrayOf(1), doubleArrayOf(-1.0)),
            SparseVector.of(2, IntArray(0), DoubleArray(0)),
        )) {
            val viaSparse = gemv(a, x)
            val viaDense = gemv(dense, x)
            assertEquals(viaDense, viaSparse, "gemv disagreed for $x")
        }
        // And against the member form, which takes the transpose flag.
        assertTrue(
            a.gemv(doubleArrayOf(2.0, -1.0)).contentEquals(gemv(a, DenseVector.of(doubleArrayOf(2.0, -1.0))).data),
        )
    }

    /**
     * Unsorted or duplicated rows within a column are rejected at construction.
     *
     * Not a style rule: [get] binary-searches the column, so a descending pair would not throw, it would
     * report a stored entry as absent — and duplicate rows would make [get] and `forEachInColumn`
     * disagree about the value at a position. The format documented ascending rows long before anything
     * checked them.
     */
    @Test
    fun `SparseMatrix rejects unsorted or duplicated rows within a column`() {
        assertFailsWith<IllegalArgumentException> {
            SparseMatrix(2, 1, intArrayOf(0, 2), intArrayOf(1, 0), doubleArrayOf(5.0, 7.0))
        }
        assertFailsWith<IllegalArgumentException> {
            SparseMatrix(2, 1, intArrayOf(0, 2), intArrayOf(1, 1), doubleArrayOf(5.0, 7.0))
        }
        // Across a column boundary the indices restart, so descending there is legitimate.
        val ok = SparseMatrix(2, 2, intArrayOf(0, 1, 2), intArrayOf(1, 0), doubleArrayOf(5.0, 7.0))
        assertEquals(5.0, ok[1, 0])
        assertEquals(7.0, ok[0, 1])
    }
}
