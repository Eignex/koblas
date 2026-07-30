package com.eignex.koblas

import kotlin.test.Test
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
}
