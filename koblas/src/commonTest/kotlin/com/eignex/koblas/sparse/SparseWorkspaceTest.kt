package com.eignex.koblas.sparse

import com.eignex.koblas.ExperimentalKoblasApi
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@OptIn(ExperimentalKoblasApi::class)
class SparseWorkspaceTest {
    @Test
    fun `scatter preserves first touch order across calls`() {
        val accumulator = DoubleArray(6)
        val marks = IntArray(6)
        val touched = IntArray(6)

        var count = SparseWorkspace.scatterAxpy(
            2.0, intArrayOf(4, 1), 0, doubleArrayOf(3.0, -2.0), 0, 2,
            accumulator, marks, 7, touched, 0, 0,
        )
        count = SparseWorkspace.scatterAxpy(
            -1.0, intArrayOf(1, 5), 0, doubleArrayOf(4.0, 8.0), 0, 2,
            accumulator, marks, 7, touched, 0, count,
        )

        assertEquals(3, count)
        assertContentEquals(intArrayOf(4, 1, 5), touched.copyOf(count))
        assertContentEquals(doubleArrayOf(0.0, -8.0, 0.0, 0.0, 6.0, -8.0), accumulator)
        assertContentEquals(intArrayOf(0, 7, 0, 0, 7, 7), marks)
    }

    @Test
    fun `scatter uses input slice offsets`() {
        val accumulator = DoubleArray(5)
        val marks = IntArray(5)
        val touched = intArrayOf(-1, -1, -1, -1, -1)

        val count = SparseWorkspace.scatterAxpy(
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

        var count = SparseWorkspace.scatterAxpy(
            1.0, indices, 0, doubleArrayOf(3.0), 0, 1,
            accumulator, marks, 4, touched, 0, 0,
        )
        count = SparseWorkspace.scatterAxpy(
            -1.0, indices, 0, doubleArrayOf(3.0), 0, 1,
            accumulator, marks, 4, touched, 0, count,
        )
        val outIndices = IntArray(4) { -1 }
        val outValues = DoubleArray(4) { 9.0 }

        val retained = SparseWorkspace.gatherTouched(
            touched,
            0,
            count,
            accumulator,
            outIndices,
            0,
            outValues,
            0,
        )
        val compacted = SparseWorkspace.gatherTouched(
            touched, 0, count, accumulator, outIndices, 0, outValues, 0, compactExactZeros = true,
        )

        assertEquals(1, count)
        assertEquals(1, retained)
        assertEquals(2, outIndices[0])
        assertEquals(0.0, outValues[0])
        assertEquals(0, compacted)
        assertEquals(4, marks[2])
    }

    @Test
    fun `zero alpha evaluates nonfinite arithmetic`() {
        val accumulator = DoubleArray(3)
        val marks = IntArray(3)
        val touched = IntArray(3)

        val count = SparseWorkspace.scatterAxpy(
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

        SparseWorkspace.scatterAxpy(
            1.0, intArrayOf(0), 0, doubleArrayOf(-0.0), 0, 1,
            accumulator, marks, 12, touched, 0, 0,
        )

        assertEquals(0.0.toBits(), accumulator[0].toBits())
    }

    @Test
    fun `gather clear compacts and clears every touched entry`() {
        val touched = intArrayOf(9, 3, 1, 4, 9)
        val accumulator = doubleArrayOf(8.0, -0.0, 7.0, Double.NaN, Double.NEGATIVE_INFINITY)
        val marks = intArrayOf(0, 6, 0, 6, 6)
        val outIndices = IntArray(5) { -1 }
        val outValues = DoubleArray(5)

        val written = SparseWorkspace.gatherClearTouched(
            touched, 1, 3, accumulator, marks,
            outIndices, 1, outValues, 1, compactExactZeros = true,
        )

        assertEquals(2, written)
        assertContentEquals(intArrayOf(-1, 3, 4, -1, -1), outIndices)
        assertTrue(outValues[1].isNaN())
        assertEquals(Double.NEGATIVE_INFINITY, outValues[2])
        assertContentEquals(doubleArrayOf(8.0, 0.0, 7.0, 0.0, 0.0), accumulator)
        assertContentEquals(IntArray(5), marks)
    }

    @Test
    fun `active maximum ignores inactive rows`() {
        val maximum = SparseWorkspace.activeColumnMaxAbs(
            intArrayOf(4, 1, 3),
            0,
            doubleArrayOf(Double.POSITIVE_INFINITY, -7.0, 5.0),
            0,
            3,
            booleanArrayOf(false, true, false, true, false),
        )

        assertEquals(7.0, maximum)
    }

    @Test
    fun `active maximum reports nonfinite and empty support`() {
        val rows = intArrayOf(0, 2)
        val values = doubleArrayOf(1.0, Double.NaN)

        val nonfinite = SparseWorkspace.activeColumnMaxAbs(rows, 0, values, 0, 2, booleanArrayOf(true, false, true))
        val empty = SparseWorkspace.activeColumnMaxAbs(rows, 0, values, 0, 2, BooleanArray(3))

        assertTrue(nonfinite.isNaN())
        assertEquals(0.0, empty)
    }

    @Test
    fun `candidate filter writes relative positions in input order`() {
        val output = IntArray(6) { -1 }

        val written = SparseWorkspace.pivotCandidatePositions(
            intArrayOf(9, 4, 1, 3, 0, 9), 1,
            doubleArrayOf(9.0, -8.0, 4.0, 0.0, 7.0, 9.0), 1,
            4, booleanArrayOf(true, true, false, true, true),
            columnMaximum = 8.0,
            absoluteTolerance = 3.0,
            relativeThreshold = 0.75,
            outPositions = output,
            outOffset = 1,
        )

        assertEquals(2, written)
        assertContentEquals(intArrayOf(-1, 0, 3, -1, -1, -1), output)
    }

    @Test
    fun `empty slices at array ends do nothing`() {
        val accumulator = DoubleArray(2)
        val marks = IntArray(2)
        val touched = IntArray(2)
        val output = IntArray(2)

        val scattered = SparseWorkspace.scatterAxpy(
            1.0, IntArray(0), 0, DoubleArray(0), 0, 0,
            accumulator, marks, 1, touched, 2, 0,
        )
        val gathered = SparseWorkspace.gatherTouched(
            touched,
            2,
            0,
            accumulator,
            output,
            2,
            DoubleArray(2),
            2,
        )
        val candidates = SparseWorkspace.pivotCandidatePositions(
            IntArray(0), 0, DoubleArray(0), 0, 0, BooleanArray(0),
            0.0, 0.0, 0.0, output, 2,
        )

        assertEquals(0, scattered)
        assertEquals(0, gathered)
        assertEquals(0, candidates)
    }

    @Test
    fun `scatter capacity failure precedes mutation`() {
        val accumulator = doubleArrayOf(1.0, 2.0)
        val marks = intArrayOf(0, 0)
        val touched = intArrayOf(9)

        assertFailsWith<IllegalArgumentException> {
            SparseWorkspace.scatterAxpy(
                1.0, intArrayOf(0, 1), 0, doubleArrayOf(3.0, 4.0), 0, 2,
                accumulator, marks, 1, touched, 0, 0,
            )
        }

        assertContentEquals(doubleArrayOf(1.0, 2.0), accumulator)
        assertContentEquals(intArrayOf(0, 0), marks)
        assertContentEquals(intArrayOf(9), touched)
    }

    @Test
    fun `scatter updates existing support at full capacity`() {
        val accumulator = doubleArrayOf(2.0, 0.0, -3.0)
        val marks = intArrayOf(5, 0, 5)
        val touched = intArrayOf(0, 2)

        val count = SparseWorkspace.scatterAxpy(
            2.0, intArrayOf(2, 0), 0, doubleArrayOf(4.0, -1.0), 0, 2,
            accumulator, marks, 5, touched, 0, 2,
        )

        assertEquals(2, count)
        assertContentEquals(doubleArrayOf(0.0, 0.0, 5.0), accumulator)
        assertContentEquals(intArrayOf(0, 2), touched)
    }

    @Test
    fun `scatter reserves capacity for actual new touches`() {
        val accumulator = doubleArrayOf(3.0, 0.0)
        val marks = intArrayOf(6, 0)
        val touched = intArrayOf(0, -1)

        val count = SparseWorkspace.scatterAxpy(
            1.0, intArrayOf(0, 1), 0, doubleArrayOf(2.0, 4.0), 0, 2,
            accumulator, marks, 6, touched, 0, 1,
        )

        assertEquals(2, count)
        assertContentEquals(doubleArrayOf(5.0, 4.0), accumulator)
        assertContentEquals(intArrayOf(0, 1), touched)
    }

    @Test
    fun `checked scatter latches nonfinite and nonzero product underflow`() {
        val accumulator = DoubleArray(2)
        val marks = IntArray(2)
        val touched = IntArray(2)
        val status = intArrayOf(0, 4)

        var count = SparseWorkspace.scatterAxpyChecked(
            Double.MIN_VALUE, intArrayOf(0), 0, doubleArrayOf(0.5), 0, 1,
            accumulator, marks, 9, touched, 0, 0, status, 1,
        )
        count = SparseWorkspace.scatterAxpyChecked(
            Double.MAX_VALUE, intArrayOf(1), 0, doubleArrayOf(2.0), 0, 1,
            accumulator, marks, 9, touched, 0, count, status, 1,
        )

        assertEquals(2, count)
        assertEquals(0.0, accumulator[0])
        assertEquals(Double.POSITIVE_INFINITY, accumulator[1])
        assertEquals(
            4 or SparseWorkspace.SCATTER_NONFINITE or SparseWorkspace.SCATTER_NONZERO_PRODUCT_UNDERFLOW,
            status[1],
        )
    }

    @Test
    fun `gather clear capacity failure precedes clearing`() {
        val accumulator = doubleArrayOf(0.0, 4.0)
        val marks = intArrayOf(0, 3)

        assertFailsWith<IllegalArgumentException> {
            SparseWorkspace.gatherClearTouched(
                intArrayOf(1), 0, 1, accumulator, marks,
                IntArray(0), 0, DoubleArray(1), 0,
            )
        }

        assertContentEquals(doubleArrayOf(0.0, 4.0), accumulator)
        assertContentEquals(intArrayOf(0, 3), marks)
    }

    @Test
    fun `overlapping integer windows are rejected before mutation`() {
        val shared = intArrayOf(0, 1, -1, -1)
        val accumulator = DoubleArray(2)
        val marks = IntArray(2)

        assertFailsWith<IllegalArgumentException> {
            SparseWorkspace.scatterAxpy(
                1.0, shared, 0, doubleArrayOf(2.0), 0, 1,
                accumulator, marks, 1, shared, 0, 0,
            )
        }

        assertContentEquals(doubleArrayOf(0.0, 0.0), accumulator)
        assertContentEquals(intArrayOf(0, 0), marks)
    }

    @Test
    fun `invalid candidate thresholds precede output writes`() {
        val output = intArrayOf(7)

        assertFailsWith<IllegalArgumentException> {
            SparseWorkspace.pivotCandidatePositions(
                intArrayOf(0), 0, doubleArrayOf(1.0), 0, 1, booleanArrayOf(true),
                Double.NaN, 0.0, 0.0, output, 0,
            )
        }

        assertContentEquals(intArrayOf(7), output)
    }

    @Test
    fun `candidate overlap is rejected before writes`() {
        val shared = intArrayOf(0, 1, 8)
        assertFailsWith<IllegalArgumentException> {
            SparseWorkspace.pivotCandidatePositions(
                shared, 0, doubleArrayOf(2.0), 0, 1, booleanArrayOf(true),
                2.0, 0.0, 0.0, shared, 0,
            )
        }

        assertContentEquals(intArrayOf(0, 1, 8), shared)
    }

    @Test
    fun `invalid candidate row is rejected before writes`() {
        val output = intArrayOf(8)
        assertFailsWith<IllegalArgumentException> {
            SparseWorkspace.pivotCandidatePositions(
                intArrayOf(2), 0, doubleArrayOf(2.0), 0, 1, booleanArrayOf(true),
                2.0, 0.0, 0.0, output, 0,
            )
        }

        assertContentEquals(intArrayOf(8), output)
    }
}
