package com.eignex.koblas.sparse

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SparsePrimitivesTest {
    @Test
    fun `scatter preserves first touch order across calls`() {
        val accumulator = DoubleArray(6)
        val marks = IntArray(6)
        val touched = IntArray(6)

        var count = SparsePrimitives.scatterWorkspace(
            2.0, intArrayOf(4, 1), 0, doubleArrayOf(3.0, -2.0), 0, 2,
            accumulator, marks, 7, touched, 0, 0,
        )
        count = SparsePrimitives.scatterWorkspace(
            -1.0, intArrayOf(1, 5), 0, doubleArrayOf(4.0, 8.0), 0, 2,
            accumulator, marks, 7, touched, 0, count,
        )

        assertEquals(3, count)
        assertContentEquals(intArrayOf(4, 1, 5), touched.copyOf(count))
        assertContentEquals(doubleArrayOf(0.0, -8.0, 0.0, 0.0, 6.0, -8.0), accumulator)
        assertContentEquals(intArrayOf(0, 7, 0, 0, 7, 7), marks)
    }

    @Test
    fun `scatter reads its own window of each input`() {
        val accumulator = DoubleArray(5)
        val marks = IntArray(5)
        val touched = intArrayOf(-1, -1, -1, -1, -1)

        val count = SparsePrimitives.scatterWorkspace(
            0.5, intArrayOf(99, 3, 0, 99), 1, doubleArrayOf(99.0, 8.0, -6.0, 99.0), 1, 2,
            accumulator, marks, 2, touched, 1, 0,
        )

        assertEquals(2, count)
        assertContentEquals(intArrayOf(-1, 3, 0, -1, -1), touched)
        assertEquals(4.0, accumulator[3])
        assertEquals(-3.0, accumulator[0])
    }

    @Test
    fun `scatter retains cancellation until compaction`() {
        val accumulator = DoubleArray(4)
        val marks = IntArray(4)
        val touched = IntArray(4)
        val indices = intArrayOf(2)

        var count = SparsePrimitives.scatterWorkspace(
            1.0, indices, 0, doubleArrayOf(3.0), 0, 1, accumulator, marks, 4, touched, 0, 0,
        )
        count = SparsePrimitives.scatterWorkspace(
            -1.0, indices, 0, doubleArrayOf(3.0), 0, 1, accumulator, marks, 4, touched, 0, count,
        )
        val outIndices = IntArray(4) { -1 }
        val outValues = DoubleArray(4) { 9.0 }

        val retained = SparsePrimitives.gatherWorkspace(
            touched, 0, count, accumulator, outIndices, 0, outValues, 0,
            compactExactZeros = false, marks = null,
        )
        val compacted = SparsePrimitives.gatherWorkspace(
            touched, 0, count, accumulator, outIndices, 0, outValues, 0,
            compactExactZeros = true, marks = null,
        )

        assertEquals(1, count)
        assertEquals(1, retained)
        assertEquals(2, outIndices[0])
        assertEquals(0.0, outValues[0])
        assertEquals(0, compacted)
        assertEquals(4, marks[2])
    }

    @Test
    fun `scatter applies a zero multiplier to nonfinite values`() {
        val accumulator = DoubleArray(3)
        val marks = IntArray(3)
        val touched = IntArray(3)

        val count = SparsePrimitives.scatterWorkspace(
            0.0, intArrayOf(0, 2), 0, doubleArrayOf(5.0, Double.POSITIVE_INFINITY), 0, 2,
            accumulator, marks, 3, touched, 0, 0,
        )

        assertEquals(2, count)
        assertEquals(0.0, accumulator[0])
        assertTrue(accumulator[2].isNaN())
        assertContentEquals(intArrayOf(0, 2), touched.copyOf(count))
    }

    @Test
    fun `first touch adds onto positive zero`() {
        val accumulator = doubleArrayOf(9.0)
        val marks = IntArray(1)
        val touched = IntArray(1)

        SparsePrimitives.scatterWorkspace(
            1.0, intArrayOf(0), 0, doubleArrayOf(-0.0), 0, 1, accumulator, marks, 12, touched, 0, 0,
        )

        assertEquals(0.0.toBits(), accumulator[0].toBits())
    }

    @Test
    fun `a zero epoch is rejected before any mutation`() {
        val accumulator = doubleArrayOf(5.0)
        val marks = IntArray(1)
        val touched = IntArray(1)

        assertFailsWith<IllegalArgumentException> {
            SparsePrimitives.scatterWorkspace(
                1.0, intArrayOf(0), 0, doubleArrayOf(2.0), 0, 1, accumulator, marks, 0, touched, 0, 0,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            SparsePrimitives.scatterWorkspaceChecked(
                1.0, intArrayOf(0), 0, doubleArrayOf(2.0), 0, 1,
                accumulator, marks, 0, touched, 0, 0, IntArray(1), 0, 1, 2,
            )
        }
        assertEquals(5.0, accumulator[0])
    }

    @Test
    fun `checked scatter latches nonfinite and nonzero product underflow`() {
        val accumulator = DoubleArray(2)
        val marks = IntArray(2)
        val touched = IntArray(2)
        val status = IntArray(1)

        SparsePrimitives.scatterWorkspaceChecked(
            1e-300, intArrayOf(0, 1), 0, doubleArrayOf(1e-300, Double.POSITIVE_INFINITY), 0, 2,
            accumulator, marks, 5, touched, 0, 0, status, 0, NONFINITE, UNDERFLOW,
        )

        assertEquals(NONFINITE or UNDERFLOW, status[0])
        assertEquals(0.0, accumulator[0])
        assertEquals(Double.POSITIVE_INFINITY, accumulator[1])
    }

    @Test
    fun `checked scatter leaves a finite exact product unflagged`() {
        val accumulator = DoubleArray(1)
        val marks = IntArray(1)
        val touched = IntArray(1)
        val status = IntArray(1)

        SparsePrimitives.scatterWorkspaceChecked(
            0.0, intArrayOf(0), 0, doubleArrayOf(4.0), 0, 1,
            accumulator, marks, 5, touched, 0, 0, status, 0, NONFINITE, UNDERFLOW,
        )

        assertEquals(0, status[0])
    }

    @Test
    fun `gather clears every touched entry when marks are supplied`() {
        val touched = intArrayOf(9, 3, 1, 4, 9)
        val accumulator = doubleArrayOf(8.0, -0.0, 7.0, Double.NaN, Double.NEGATIVE_INFINITY)
        val marks = intArrayOf(0, 6, 0, 6, 6)
        val outIndices = IntArray(5) { -1 }
        val outValues = DoubleArray(5)

        val written = SparsePrimitives.gatherWorkspace(
            touched, 1, 3, accumulator, outIndices, 1, outValues, 1,
            compactExactZeros = true, marks = marks,
        )

        assertEquals(2, written)
        assertContentEquals(intArrayOf(-1, 3, 4, -1, -1), outIndices)
        assertTrue(outValues[1].isNaN())
        assertEquals(Double.NEGATIVE_INFINITY, outValues[2])
        assertContentEquals(doubleArrayOf(8.0, 0.0, 7.0, 0.0, 0.0), accumulator)
        assertContentEquals(IntArray(5), marks)
    }

    @Test
    fun `gather without marks leaves the accumulator loaded`() {
        val touched = intArrayOf(2, 0)
        val accumulator = doubleArrayOf(-4.0, 9.0, 3.0)
        val marks = intArrayOf(6, 0, 6)
        val outIndices = IntArray(2)
        val outValues = DoubleArray(2)

        SparsePrimitives.gatherWorkspace(
            touched, 0, 2, accumulator, outIndices, 0, outValues, 0,
            compactExactZeros = false, marks = null,
        )

        assertContentEquals(intArrayOf(2, 0), outIndices)
        assertContentEquals(doubleArrayOf(3.0, -4.0), outValues)
        assertContentEquals(doubleArrayOf(-4.0, 9.0, 3.0), accumulator)
        assertContentEquals(intArrayOf(6, 0, 6), marks)
    }

    @Test
    fun `clear resets the touched entries and leaves the rest alone`() {
        val touched = intArrayOf(7, 3, 0, 7)
        val accumulator = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val marks = intArrayOf(5, 5, 5, 5)

        SparsePrimitives.clearWorkspace(touched, 1, 2, accumulator, marks)

        assertContentEquals(doubleArrayOf(0.0, 2.0, 3.0, 0.0), accumulator)
        assertContentEquals(intArrayOf(0, 5, 5, 0), marks)
        assertContentEquals(intArrayOf(7, 3, 0, 7), touched)
    }

    @Test
    fun `clear writes positive zero over a negative accumulator entry`() {
        val accumulator = doubleArrayOf(-0.0)
        val marks = intArrayOf(3)

        SparsePrimitives.clearWorkspace(intArrayOf(0), 0, 1, accumulator, marks)

        assertEquals(0.0.toBits(), accumulator[0].toBits())
        assertEquals(0, marks[0])
    }

    @Test
    fun `active maximum ignores inactive rows`() {
        val maximum = SparsePrimitives.activeMaximum(
            intArrayOf(4, 1, 3),
            0,
            doubleArrayOf(Double.POSITIVE_INFINITY, -7.0, 5.0),
            0,
            3,
            BooleanArray(5) { it != 4 },
        )

        assertEquals(7.0, maximum)
    }

    @Test
    fun `active maximum reports nonfinite and empty support`() {
        val active = BooleanArray(3) { true }
        val nonfinite = SparsePrimitives.activeMaximum(
            intArrayOf(0, 2),
            0,
            doubleArrayOf(3.0, Double.NaN),
            0,
            2,
            active,
        )
        val empty = SparsePrimitives.activeMaximum(
            intArrayOf(0, 2),
            0,
            doubleArrayOf(3.0, 4.0),
            0,
            2,
            BooleanArray(3),
        )

        assertTrue(nonfinite.isNaN())
        assertEquals(0.0, empty)
    }

    @Test
    fun `candidate filter writes relative positions in input order`() {
        val out = IntArray(4) { -1 }

        val written = SparsePrimitives.selectPivotCandidates(
            intArrayOf(0, 1, 2, 3), 0, doubleArrayOf(0.5, 0.0, -9.0, Double.NaN), 0, 4,
            BooleanArray(4) { it != 1 }, absoluteTolerance = 0.25, relativeCutoff = 0.4,
            outPositions = out, outOffset = 0,
        )

        assertEquals(2, written)
        assertContentEquals(intArrayOf(0, 2, -1, -1), out)
    }

    @Test
    fun `a nonfinite cutoff is rejected before any candidate is written`() {
        val out = IntArray(1) { -1 }

        assertFailsWith<IllegalArgumentException> {
            SparsePrimitives.selectPivotCandidates(
                intArrayOf(0), 0, doubleArrayOf(1.0), 0, 1, BooleanArray(1) { true },
                absoluteTolerance = 0.0, relativeCutoff = Double.NaN, outPositions = out, outOffset = 0,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            SparsePrimitives.selectPivotCandidates(
                intArrayOf(0), 0, doubleArrayOf(1.0), 0, 1, BooleanArray(1) { true },
                absoluteTolerance = -1.0, relativeCutoff = 0.0, outPositions = out, outOffset = 0,
            )
        }
        assertEquals(-1, out[0])
    }

    @Test
    fun `empty slices at array ends do nothing`() {
        val accumulator = doubleArrayOf(1.0)
        val marks = intArrayOf(2)
        val touched = intArrayOf(0)

        val scattered = SparsePrimitives.scatterWorkspace(
            1.0, IntArray(0), 0, DoubleArray(0), 0, 0, accumulator, marks, 2, touched, 1, 0,
        )
        val gathered = SparsePrimitives.gatherWorkspace(
            touched, 1, 0, accumulator, IntArray(0), 0, DoubleArray(0), 0,
            compactExactZeros = true, marks = marks,
        )
        SparsePrimitives.clearWorkspace(touched, 1, 0, accumulator, marks)

        assertEquals(0, scattered)
        assertEquals(0, gathered)
        assertContentEquals(doubleArrayOf(1.0), accumulator)
        assertContentEquals(intArrayOf(2), marks)
    }

    private companion object {
        const val NONFINITE = 1
        const val UNDERFLOW = 2
    }
}
