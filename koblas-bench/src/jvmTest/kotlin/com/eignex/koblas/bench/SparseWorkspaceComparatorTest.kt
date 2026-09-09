package com.eignex.koblas.bench

import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.sparse.SparseWorkspace
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@OptIn(ExperimentalKoblasApi::class)
class SparseWorkspaceComparatorTest {
    @Test
    fun `checked baseline agrees across offsets diagnostics and epoch state`() {
        val indices = intArrayOf(99, 5, 2, 8, 77)
        val values = doubleArrayOf(99.0, 4.0, Double.POSITIVE_INFINITY, Double.MIN_VALUE, 77.0)
        val expectedAccumulator = DoubleArray(12) { 41.0 }
        val actualAccumulator = expectedAccumulator.copyOf()
        val expectedMarks = IntArray(12).also { it[2] = 7 }
        val actualMarks = expectedMarks.copyOf()
        val expectedTouched = IntArray(8) { -1 }.also { it[2] = 2 }
        val actualTouched = expectedTouched.copyOf()
        val expectedStatus = intArrayOf(13, 4, 17)
        val actualStatus = expectedStatus.copyOf()

        val expectedCount = SparseWorkspace.scatterAxpyChecked(
            Double.MIN_VALUE, indices, 1, values, 1, 3,
            expectedAccumulator, expectedMarks, 7, expectedTouched, 2, 1, expectedStatus, 1,
        )
        val actualCount = SparseWorkspaceComparators.scatterAxpyCheckedBaseline(
            Double.MIN_VALUE, indices, 1, values, 1, 3,
            actualAccumulator, actualMarks, 7, actualTouched, 2, 1, actualStatus, 1,
        )

        assertEquals(expectedCount, actualCount)
        assertContentEquals(expectedAccumulator, actualAccumulator)
        assertContentEquals(expectedMarks, actualMarks)
        assertContentEquals(expectedTouched, actualTouched)
        assertContentEquals(expectedStatus, actualStatus)
    }

    @Test
    fun `reduction baselines agree for offsets activity thresholds and nonfinite values`() {
        val rows = intArrayOf(99, 5, 2, 7, 1, 88)
        val values = doubleArrayOf(99.0, -0.0, -4.0, Double.NaN, 1.5, 88.0)
        val active = BooleanArray(10).also {
            it[1] = true
            it[2] = true
            it[5] = true
        }
        val expectedPositions = IntArray(8) { -1 }
        val actualPositions = expectedPositions.copyOf()

        val expectedMaximum = SparseWorkspace.activeColumnMaxAbs(rows, 1, values, 1, 4, active)
        val actualMaximum = SparseWorkspaceComparators.activeColumnMaxAbsBaseline(rows, 1, values, 1, 4, active)
        val expectedCount = SparseWorkspace.pivotCandidatePositions(
            rows, 1, values, 1, 4, active, expectedMaximum, 1.0, 0.25, expectedPositions, 2,
        )
        val actualCount = SparseWorkspaceComparators.pivotCandidatePositionsBaseline(
            rows, 1, values, 1, 4, active, actualMaximum, 1.0, 0.25, actualPositions, 2,
        )

        assertEquals(expectedMaximum, actualMaximum)
        assertEquals(expectedCount, actualCount)
        assertContentEquals(expectedPositions, actualPositions)

        active[7] = true
        assertTrue(SparseWorkspace.activeColumnMaxAbs(rows, 1, values, 1, 4, active).isNaN())
        assertTrue(SparseWorkspaceComparators.activeColumnMaxAbsBaseline(rows, 1, values, 1, 4, active).isNaN())
    }

    @Test
    fun `baseline capacity failure occurs before mutation`() {
        val accumulator = doubleArrayOf(8.0, 9.0, 10.0)
        val marks = intArrayOf(0, 0, 0)
        val touched = intArrayOf(77)
        val status = intArrayOf(5)

        assertFailsWith<IllegalArgumentException> {
            SparseWorkspaceComparators.scatterAxpyCheckedBaseline(
                2.0, intArrayOf(0, 2), 0, doubleArrayOf(3.0, 4.0), 0, 2,
                accumulator, marks, 4, touched, 0, 0, status, 0,
            )
        }

        assertContentEquals(doubleArrayOf(8.0, 9.0, 10.0), accumulator)
        assertContentEquals(intArrayOf(0, 0, 0), marks)
        assertContentEquals(intArrayOf(77), touched)
        assertContentEquals(intArrayOf(5), status)
    }

    @Test
    fun `checked baseline evaluates alpha zero and accepts empty offset slices`() {
        val expectedAccumulator = doubleArrayOf(5.0, -7.0)
        val actualAccumulator = expectedAccumulator.copyOf()
        val expectedMarks = intArrayOf(0, 0)
        val actualMarks = expectedMarks.copyOf()
        val expectedTouched = intArrayOf(-1, -1, -1)
        val actualTouched = expectedTouched.copyOf()
        val expectedStatus = intArrayOf(9, 0)
        val actualStatus = expectedStatus.copyOf()

        val expectedCount = SparseWorkspace.scatterAxpyChecked(
            0.0, intArrayOf(99, 1), 1, doubleArrayOf(99.0, Double.POSITIVE_INFINITY), 1, 1,
            expectedAccumulator, expectedMarks, 3, expectedTouched, 1, 0, expectedStatus, 1,
        )
        val actualCount = SparseWorkspaceComparators.scatterAxpyCheckedBaseline(
            0.0, intArrayOf(99, 1), 1, doubleArrayOf(99.0, Double.POSITIVE_INFINITY), 1, 1,
            actualAccumulator, actualMarks, 3, actualTouched, 1, 0, actualStatus, 1,
        )
        assertEquals(expectedCount, actualCount)
        assertContentEquals(expectedAccumulator, actualAccumulator)
        assertContentEquals(expectedMarks, actualMarks)
        assertContentEquals(expectedTouched, actualTouched)
        assertContentEquals(expectedStatus, actualStatus)

        val emptyExpected = SparseWorkspace.scatterAxpyChecked(
            2.0, intArrayOf(99), 1, doubleArrayOf(99.0), 1, 0,
            expectedAccumulator, expectedMarks, 3, expectedTouched, 2, expectedCount, expectedStatus, 1,
        )
        val emptyActual = SparseWorkspaceComparators.scatterAxpyCheckedBaseline(
            2.0, intArrayOf(99), 1, doubleArrayOf(99.0), 1, 0,
            actualAccumulator, actualMarks, 3, actualTouched, 2, actualCount, actualStatus, 1,
        )
        assertEquals(emptyExpected, emptyActual)
    }

    @Test
    fun `composed scatter agrees for first touch overlap and offsets`() {
        val comparator = indexedComparator()
        val indices = intArrayOf(91, 7, 2, 9, 4, 82)
        val values = doubleArrayOf(91.0, 1.5, -2.0, 3.25, -4.5, 82.0)
        val expectedAccumulator = DoubleArray(12) { 123.0 }.also {
            it[2] = 8.0
            it[4] = -1.0
        }
        val actualAccumulator = expectedAccumulator.copyOf()
        val expectedMarks = IntArray(12).also {
            it[2] = 6
            it[4] = 6
        }
        val actualMarks = expectedMarks.copyOf()
        val expectedTouched = IntArray(10) { -1 }.also {
            it[2] = 2
            it[3] = 4
        }
        val actualTouched = expectedTouched.copyOf()

        val expectedCount = SparseWorkspace.scatterAxpy(
            0.75, indices, 1, values, 1, 4,
            expectedAccumulator, expectedMarks, 6, expectedTouched, 2, 2,
        )
        val actualCount = SparseWorkspaceComparators.scatterAxpyOneMkl(
            comparator, 0.75, indices, 1, values, 1, 4,
            actualAccumulator, actualMarks, 6, actualTouched, 2, 2,
        )

        assertEquals(expectedCount, actualCount)
        assertContentEquals(expectedAccumulator, actualAccumulator)
        assertContentEquals(expectedMarks, actualMarks)
        assertContentEquals(expectedTouched, actualTouched)

        comparator.indexedAxpy(2.0, values, 1, indices, 1, 0, actualAccumulator)
        assertContentEquals(expectedAccumulator, actualAccumulator)
    }

    @Test
    fun `composed repeated short scatters preserve growing first touch order`() {
        val comparator = indexedComparator()
        val calls = arrayOf(
            intArrayOf(7, 2, 9, 4),
            intArrayOf(2, 4, 6, 1),
            intArrayOf(9, 1, 8, 5),
        )
        val callValues = arrayOf(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0),
            doubleArrayOf(-2.0, -4.0, 5.0, 6.0),
            doubleArrayOf(-3.0, -6.0, 7.0, 8.0),
        )
        val expectedAccumulator = DoubleArray(12) { 91.0 }
        val actualAccumulator = expectedAccumulator.copyOf()
        val expectedMarks = IntArray(12)
        val actualMarks = expectedMarks.copyOf()
        val expectedTouched = IntArray(12) { -1 }
        val actualTouched = expectedTouched.copyOf()
        var expectedCount = 0
        var actualCount = 0
        for (call in calls.indices) {
            expectedCount = SparseWorkspace.scatterAxpy(
                1.0, calls[call], 0, callValues[call], 0, 4,
                expectedAccumulator, expectedMarks, 8, expectedTouched, 1, expectedCount,
            )
            actualCount = SparseWorkspaceComparators.scatterAxpyOneMkl(
                comparator, 1.0, calls[call], 0, callValues[call], 0, 4,
                actualAccumulator, actualMarks, 8, actualTouched, 1, actualCount,
            )
        }
        assertEquals(expectedCount, actualCount)
        assertContentEquals(expectedAccumulator, actualAccumulator)
        assertContentEquals(expectedMarks, actualMarks)
        assertContentEquals(expectedTouched, actualTouched)
    }

    @Test
    fun `composed gathers agree for order compaction clearing and offsets`() {
        val comparator = indexedComparator()
        val touched = intArrayOf(77, 6, 1, 8, 3, 66)
        val source = DoubleArray(10) { 19.0 }.also {
            it[6] = 2.5
            it[1] = 0.0
            it[8] = Double.NEGATIVE_INFINITY
            it[3] = -0.0
        }
        for (compact in listOf(false, true)) {
            val expectedIndices = IntArray(8) { -1 }
            val actualIndices = expectedIndices.copyOf()
            val expectedValues = DoubleArray(8) { 17.0 }
            val actualValues = expectedValues.copyOf()
            val scratchValues = DoubleArray(8) { 23.0 }
            val expectedCount = SparseWorkspace.gatherTouched(
                touched, 1, 4, source, expectedIndices, 2, expectedValues, 2, compact,
            )
            val actualCount = SparseWorkspaceComparators.gatherTouchedOneMkl(
                comparator, touched, 1, 4, source, scratchValues, 1,
                actualIndices, 2, actualValues, 2, compact,
            )
            assertEquals(expectedCount, actualCount, "compact=$compact")
            assertContentEquals(expectedIndices, actualIndices, "compact=$compact")
            assertContentEquals(expectedValues, actualValues, "compact=$compact")

            repeat(2) { invocation ->
                val expectedAccumulator = source.copyOf()
                val actualAccumulator = source.copyOf()
                val expectedMarks = IntArray(10) { 9 }
                val actualMarks = expectedMarks.copyOf()
                expectedIndices.fill(-1)
                actualIndices.fill(-1)
                expectedValues.fill(17.0)
                actualValues.fill(17.0)
                val expectedClearCount = SparseWorkspace.gatherClearTouched(
                    touched, 1, 4, expectedAccumulator, expectedMarks,
                    expectedIndices, 2, expectedValues, 2, compact,
                )
                val actualClearCount = SparseWorkspaceComparators.gatherClearTouchedOneMkl(
                    comparator, touched, 1, 4, actualAccumulator, actualMarks, scratchValues, 1,
                    actualIndices, 2, actualValues, 2, compact,
                )
                assertEquals(expectedClearCount, actualClearCount, "compact=$compact invocation=$invocation")
                assertContentEquals(expectedAccumulator, actualAccumulator, "accumulator")
                assertContentEquals(expectedMarks, actualMarks, "marks")
                assertContentEquals(expectedIndices, actualIndices, "indices")
                assertContentEquals(expectedValues, actualValues, "values")
            }
        }
    }
}

private fun indexedComparator(): IndexedSparseLevel1Comparator {
    val oneMkl = oneMklSparseComparator()
    if (System.getProperty("koblas.oneMklTests") == "true") {
        return checkNotNull(oneMkl) { "-Pkoblas.oneMklTests=true requires a loader-visible oneMKL runtime" }
    }
    return oneMkl ?: KotlinIndexedComparator
}

private object KotlinIndexedComparator : IndexedSparseLevel1Comparator {
    override fun indexedAxpy(
        alpha: Double,
        values: DoubleArray,
        valueOffset: Int,
        indices: IntArray,
        indexOffset: Int,
        count: Int,
        accumulator: DoubleArray,
    ) {
        for (k in 0 until count) accumulator[indices[indexOffset + k]] += alpha * values[valueOffset + k]
    }

    override fun indexedGather(
        indices: IntArray,
        indexOffset: Int,
        count: Int,
        accumulator: DoubleArray,
        outValues: DoubleArray,
        outValueOffset: Int,
    ) {
        for (k in 0 until count) outValues[outValueOffset + k] = accumulator[indices[indexOffset + k]]
    }

    override fun indexedGatherZero(
        indices: IntArray,
        indexOffset: Int,
        count: Int,
        accumulator: DoubleArray,
        outValues: DoubleArray,
        outValueOffset: Int,
    ) {
        for (k in 0 until count) {
            val index = indices[indexOffset + k]
            outValues[outValueOffset + k] = accumulator[index]
            accumulator[index] = 0.0
        }
    }
}
