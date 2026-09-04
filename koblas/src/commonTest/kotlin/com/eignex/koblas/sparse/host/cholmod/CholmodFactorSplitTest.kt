package com.eignex.koblas.sparse.host.cholmod

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertNotSame

/**
 * The factor split both CHOLMOD bindings share, over the plain arrays an extraction leaves behind. Reading
 * these off a live factor needs the library installed; the arithmetic on them does not.
 */
class CholmodFactorSplitTest {

    /** Three columns, each storing its diagonal first, which is the shape CHOLMOD converts a factor into. */
    private fun factors() = CholmodFactors(
        order = 3,
        colPtr = intArrayOf(0, 3, 5, 6),
        rowIdx = intArrayOf(0, 1, 2, 1, 2, 2),
        values = doubleArrayOf(4.0, 0.5, 0.25, 9.0, 0.75, 16.0),
        permutation = intArrayOf(2, 0, 1),
    )

    @Test
    fun `an L L transpose factor keeps the diagonal it stores`() {
        val l = factors().lower(isLl = true)

        assertContentEquals(intArrayOf(0, 3, 5, 6), l.colPtr)
        assertContentEquals(intArrayOf(0, 1, 2, 1, 2, 2), l.rowIdx)
        assertContentEquals(doubleArrayOf(4.0, 0.5, 0.25, 9.0, 0.75, 16.0), l.values)
    }

    @Test
    fun `an L D L transpose factor drops the diagonal into D`() {
        val held = factors()

        val l = held.lower(isLl = false)

        assertContentEquals(intArrayOf(0, 2, 3, 3), l.colPtr, "one entry fewer in every column")
        assertContentEquals(intArrayOf(1, 2, 2), l.rowIdx)
        assertContentEquals(doubleArrayOf(0.5, 0.25, 0.75), l.values)
        assertContentEquals(doubleArrayOf(4.0, 9.0, 16.0), held.diagonal(), "the dropped entries are D")
    }

    /** The factor outlives the read, so what it hands out cannot be the arrays it keeps. */
    @Test
    fun `the kept factor is copied rather than shared`() {
        val held = factors()

        val l = held.lower(isLl = true)

        assertNotSame(held.values, l.values)
        assertNotSame(held.colPtr, l.colPtr)
        assertNotSame(held.rowIdx, l.rowIdx)
    }
}
