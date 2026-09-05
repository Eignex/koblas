package com.eignex.koblas.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Overlap decides whether a strided product is allowed to run at all, and it answers most cases without
 * walking the entries. Every shortcut has to agree with the walk it replaces, so the cases below are checked
 * against a plain scan of the physical indices each view addresses.
 */
class F64StridedViewOverlapTest {

    private val buffer = DoubleArray(64)

    @Test
    fun `views over different buffers never overlap`() {
        val left = F64StridedMatrixView(4, 4, buffer, 0, 8)
        val right = F64StridedMatrixView(4, 4, DoubleArray(64), 0, 8)

        assertFalse(left.overlaps(right))
    }

    @Test
    fun `an empty view overlaps nothing`() {
        val empty = F64StridedMatrixView(0, 4, buffer, 0, 8)
        val whole = F64StridedMatrixView(8, 8, buffer, 0, 8)

        assertFalse(empty.overlaps(whole))
        assertFalse(whole.overlaps(empty))
    }

    @Test
    fun `column panels of one matrix do not overlap`() {
        val whole = F64StridedMatrixView(8, 8, buffer, 0, 8)
        val left = whole.view(0, 8, 0, 3)
        val right = whole.view(0, 8, 3, 5)

        assertFalse(left.overlaps(right))
        assertFalse(right.overlaps(left))
    }

    @Test
    fun `row panels of one matrix do not overlap`() {
        val whole = F64StridedMatrixView(8, 8, buffer, 0, 8)
        val top = whole.view(0, 3, 0, 8)
        val bottom = whole.view(3, 5, 0, 8)

        assertFalse(top.overlaps(bottom))
        assertFalse(bottom.overlaps(top))
    }

    @Test
    fun `blocks sharing a corner overlap`() {
        val whole = F64StridedMatrixView(8, 8, buffer, 0, 8)
        val upperLeft = whole.view(0, 5, 0, 5)
        val lowerRight = whole.view(4, 4, 4, 4)

        assertTrue(upperLeft.overlaps(lowerRight))
        assertTrue(lowerRight.overlaps(upperLeft))
    }

    /** The shortcuts have to answer the same as the walk over every pair of blocks one buffer can hold. */
    @Test
    fun `every pair of blocks agrees with a scan of the entries they address`() {
        val whole = F64StridedMatrixView(8, 8, buffer, 0, 8)
        val blocks = ArrayList<F64StridedMatrixView>()
        for (row in 0 until 8 step 3) {
            for (column in 0 until 8 step 3) {
                for (height in 1..(8 - row) step 3) {
                    for (width in 1..(8 - column) step 3) {
                        blocks.add(whole.view(row, height, column, width))
                    }
                }
            }
        }

        for (left in blocks) {
            for (right in blocks) {
                assertEquals(
                    indicesOf(left).any { it in indicesOf(right) },
                    left.overlaps(right),
                    "$left against $right",
                )
            }
        }
    }

    /** Views with different leading dimensions share no grid, so they take the entry walk. */
    @Test
    fun `views with unequal leading dimensions agree with a scan`() {
        val rng = Random(20260941)
        repeat(200) {
            val left = randomView(rng, 8)
            val right = randomView(rng, 5)

            assertEquals(
                indicesOf(left).any { it in indicesOf(right) },
                left.overlaps(right),
                "$left against $right",
            )
        }
    }

    @Test
    fun `a vector overlaps the matrix rows it runs through`() {
        val whole = F64StridedMatrixView(8, 8, buffer, 0, 8)
        val block = whole.view(0, 4, 0, 4)
        val insideRow = F64StridedVectorView(buffer, 1, 4, 8)
        val elsewhere = F64StridedVectorView(buffer, 36, 4, 8)

        assertTrue(block.overlaps(insideRow))
        assertFalse(block.overlaps(elsewhere))
    }

    @Test
    fun `a negatively strided vector is measured over the entries it reaches`() {
        val whole = F64StridedMatrixView(8, 8, buffer, 0, 8)
        val block = whole.view(0, 4, 4, 4)
        // 35 down to 32, which is column 4 of the block read from the bottom up.
        val descending = F64StridedVectorView(buffer, 35, 4, -1)

        assertTrue(block.overlaps(descending))
        // 39 down to 15, which passes the block's span without landing in any of its rows.
        assertFalse(block.overlaps(F64StridedVectorView(buffer, 39, 4, -8)))
    }

    private fun randomView(rng: Random, leadingDimension: Int): F64StridedMatrixView {
        val rows = rng.nextInt(1, leadingDimension + 1)
        val cols = rng.nextInt(1, 5)
        val span = (cols - 1) * leadingDimension + rows
        return F64StridedMatrixView(rows, cols, buffer, rng.nextInt(0, buffer.size - span + 1), leadingDimension)
    }

    private fun indicesOf(view: F64StridedMatrixView): Set<Int> {
        val out = HashSet<Int>()
        for (j in 0 until view.cols) {
            for (i in 0 until view.rows) out.add(view.offset + i + j * view.leadingDimension)
        }
        return out
    }
}
