package com.eignex.koblas.sparse.internal

import com.eignex.koblas.UnsafeKoblasApi
import kotlin.test.*

@OptIn(UnsafeKoblasApi::class)
class SparseTransposeTest {

    /**
     * A library hands back a column whose rows do not ascend, which is the form [transposeRaw] exists to
     * admit. Its result is checked rather than trusted, so a pattern koblas cannot vouch for is still the
     * one that gets scanned.
     */
    @Test
    fun `a raw transpose sorts a column a library left unordered`() {
        val transposed = transposeRaw(
            2,
            2,
            intArrayOf(0, 2, 4),
            intArrayOf(1, 0, 1, 0),
            doubleArrayOf(3.0, 1.0, 4.0, 2.0),
        )

        assertContentEquals(intArrayOf(0, 2, 4), transposed.colPtr)
        assertContentEquals(intArrayOf(0, 1, 0, 1), transposed.rowIdx)
        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0, 4.0), transposed.values)
    }

    /** Transposing twice returns the orientation it started in, with the rows now ascending. */
    @Test
    fun `sortedCsc orders the rows within every column`() {
        val sorted = sortedCsc(2, 2, intArrayOf(0, 2, 4), intArrayOf(1, 0, 1, 0), doubleArrayOf(3.0, 1.0, 4.0, 2.0))

        assertContentEquals(intArrayOf(0, 2, 4), sorted.colPtr)
        assertContentEquals(intArrayOf(0, 1, 0, 1), sorted.rowIdx)
        assertContentEquals(doubleArrayOf(1.0, 3.0, 2.0, 4.0), sorted.values)
    }

    /** The row form a library reports is the transpose already, so one pass both converts it and sorts it. */
    @Test
    fun `sortedCsc keeps a single transpose for a row form`() {
        val rowForm = sortedCsc(
            2,
            2,
            intArrayOf(0, 2, 4),
            intArrayOf(1, 0, 1, 0),
            doubleArrayOf(3.0, 1.0, 4.0, 2.0),
            transposed = true,
        )

        assertContentEquals(intArrayOf(0, 1, 0, 1), rowForm.rowIdx)
    }
}
