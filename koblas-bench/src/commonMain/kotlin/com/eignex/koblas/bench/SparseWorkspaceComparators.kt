package com.eignex.koblas.bench

import kotlin.math.abs

/** Benchmark-owned implementations of the complete workspace contracts compared by the sparse suites. */
internal object SparseWorkspaceComparators {
    /**
     * Composes oneMKL indexed AXPY with the first-touch bookkeeping required by the workspace contract.
     *
     * Timed comparisons use finite values and a finite, nonzero alpha. oneMKL is not used for the checked variant,
     * because its call does not expose per-product diagnostics and may optimize exceptional IEEE arithmetic.
     */
    @Suppress("LongParameterList")
    fun scatterAxpyOneMkl(
        comparator: IndexedSparseLevel1Comparator,
        alpha: Double,
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        accumulator: DoubleArray,
        marks: IntArray,
        epoch: Int,
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
    ): Int {
        validateScatter(
            indices, indexOffset, values, valueOffset, count,
            accumulator, marks, epoch, touched, touchedOffset, touchedCount,
        )
        var total = touchedCount
        for (k in 0 until count) {
            val index = indices[indexOffset + k]
            if (marks[index] != epoch) {
                accumulator[index] = 0.0
                marks[index] = epoch
                touched[touchedOffset + total] = index
                total++
            }
        }
        comparator.indexedAxpy(alpha, values, valueOffset, indices, indexOffset, count, accumulator)
        return total
    }

    @Suppress("LongParameterList")
    fun gatherTouchedOneMkl(
        comparator: IndexedSparseLevel1Comparator,
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        accumulator: DoubleArray,
        scratchValues: DoubleArray,
        scratchOffset: Int,
        outIndices: IntArray,
        outIndexOffset: Int,
        outValues: DoubleArray,
        outValueOffset: Int,
        compactExactZeros: Boolean,
    ): Int {
        validateGather(
            touched, touchedOffset, touchedCount, accumulator,
            outIndices, outIndexOffset, outValues, outValueOffset,
        )
        requireWindow(scratchValues.size, scratchOffset, touchedCount, "gather scratch")
        requireDistinct(accumulator, scratchValues, "accumulator and gather scratch")
        requireDistinct(scratchValues, outValues, "gather scratch and output values")
        comparator.indexedGather(touched, touchedOffset, touchedCount, accumulator, scratchValues, scratchOffset)
        return emitOrderedIndices(
            touched, touchedOffset, touchedCount, scratchValues, scratchOffset,
            outIndices, outIndexOffset, outValues, outValueOffset,
            compactExactZeros,
        )
    }

    @Suppress("LongParameterList")
    fun gatherClearTouchedOneMkl(
        comparator: IndexedSparseLevel1Comparator,
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        accumulator: DoubleArray,
        marks: IntArray,
        scratchValues: DoubleArray,
        scratchOffset: Int,
        outIndices: IntArray,
        outIndexOffset: Int,
        outValues: DoubleArray,
        outValueOffset: Int,
        compactExactZeros: Boolean,
    ): Int {
        require(accumulator.size == marks.size) { "accumulator and marks lengths differ" }
        validateGather(
            touched, touchedOffset, touchedCount, accumulator,
            outIndices, outIndexOffset, outValues, outValueOffset,
        )
        requireDistinct(marks, touched, "marks and touched")
        requireDistinct(marks, outIndices, "marks and output indices")
        requireWindow(scratchValues.size, scratchOffset, touchedCount, "gather scratch")
        requireDistinct(accumulator, scratchValues, "accumulator and gather scratch")
        requireDistinct(scratchValues, outValues, "gather scratch and output values")
        comparator.indexedGatherZero(touched, touchedOffset, touchedCount, accumulator, scratchValues, scratchOffset)
        for (k in 0 until touchedCount) marks[touched[touchedOffset + k]] = 0
        return emitOrderedIndices(
            touched, touchedOffset, touchedCount, scratchValues, scratchOffset,
            outIndices, outIndexOffset, outValues, outValueOffset,
            compactExactZeros,
        )
    }

    @Suppress("LongParameterList")
    fun scatterAxpyCheckedBaseline(
        alpha: Double,
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        accumulator: DoubleArray,
        marks: IntArray,
        epoch: Int,
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        arithmeticStatus: IntArray,
        statusOffset: Int,
    ): Int {
        requireWindow(arithmeticStatus.size, statusOffset, 1, "arithmetic status")
        requireDistinct(arithmeticStatus, indices, "arithmetic status and indices")
        requireDistinct(arithmeticStatus, marks, "arithmetic status and marks")
        requireDistinct(arithmeticStatus, touched, "arithmetic status and touched")
        validateScatter(
            indices, indexOffset, values, valueOffset, count,
            accumulator, marks, epoch, touched, touchedOffset, touchedCount,
        )
        var flags = arithmeticStatus[statusOffset]
        var total = touchedCount
        for (k in 0 until count) {
            val index = indices[indexOffset + k]
            if (marks[index] != epoch) {
                accumulator[index] = 0.0
                marks[index] = epoch
                touched[touchedOffset + total] = index
                total++
            }
            val value = values[valueOffset + k]
            val product = alpha * value
            val updated = accumulator[index] + product
            if (!product.isFinite() || !updated.isFinite()) flags = flags or 1
            if (alpha != 0.0 && value != 0.0 && product == 0.0) flags = flags or 2
            accumulator[index] = updated
        }
        arithmeticStatus[statusOffset] = flags
        return total
    }

    fun activeColumnMaxAbsBaseline(
        rowIndices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        activeRows: BooleanArray,
    ): Double {
        requireWindow(rowIndices.size, indexOffset, count, "row indices")
        requireWindow(values.size, valueOffset, count, "values")
        validateIndices(rowIndices, indexOffset, count, activeRows.size, "row indices")
        var maximum = 0.0
        for (k in 0 until count) {
            if (!activeRows[rowIndices[indexOffset + k]]) continue
            val value = values[valueOffset + k]
            if (!value.isFinite()) return Double.NaN
            maximum = maxOf(maximum, abs(value))
        }
        return maximum
    }

    @Suppress("LongParameterList")
    fun pivotCandidatePositionsBaseline(
        rowIndices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        activeRows: BooleanArray,
        columnMaximum: Double,
        absoluteTolerance: Double,
        relativeThreshold: Double,
        outPositions: IntArray,
        outOffset: Int,
    ): Int {
        require(columnMaximum.isFinite() && columnMaximum >= 0.0) { "invalid column maximum" }
        require(absoluteTolerance.isFinite() && absoluteTolerance >= 0.0) { "invalid absolute tolerance" }
        require(relativeThreshold.isFinite() && relativeThreshold >= 0.0) { "invalid relative threshold" }
        requireWindow(rowIndices.size, indexOffset, count, "row indices")
        requireWindow(values.size, valueOffset, count, "values")
        requireWindow(outPositions.size, outOffset, count, "candidate output")
        requireNonoverlap(rowIndices, indexOffset, count, outPositions, outOffset, count, "row indices and candidates")
        validateIndices(rowIndices, indexOffset, count, activeRows.size, "row indices")
        val relativeCutoff = relativeThreshold * columnMaximum
        var written = 0
        for (k in 0 until count) {
            if (!activeRows[rowIndices[indexOffset + k]]) continue
            val value = values[valueOffset + k]
            if (!value.isFinite() || value == 0.0) continue
            val magnitude = abs(value)
            if (magnitude >= absoluteTolerance && magnitude >= relativeCutoff) {
                outPositions[outOffset + written] = k
                written++
            }
        }
        return written
    }

    @Suppress("LongParameterList")
    private fun emitOrderedIndices(
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        gatheredValues: DoubleArray,
        gatheredValueOffset: Int,
        outIndices: IntArray,
        outIndexOffset: Int,
        outValues: DoubleArray,
        outValueOffset: Int,
        compactExactZeros: Boolean,
    ): Int {
        var written = 0
        for (k in 0 until touchedCount) {
            val value = gatheredValues[gatheredValueOffset + k]
            if (compactExactZeros && value == 0.0) continue
            outIndices[outIndexOffset + written] = touched[touchedOffset + k]
            outValues[outValueOffset + written] = value
            written++
        }
        return written
    }
}

@Suppress("LongParameterList")
private fun validateScatter(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    accumulator: DoubleArray,
    marks: IntArray,
    epoch: Int,
    touched: IntArray,
    touchedOffset: Int,
    touchedCount: Int,
) {
    require(epoch != 0) { "scatter epoch must be nonzero" }
    require(accumulator.size == marks.size) { "accumulator and marks lengths differ" }
    requireWindow(indices.size, indexOffset, count, "indices")
    requireWindow(values.size, valueOffset, count, "values")
    requireWindow(touched.size, touchedOffset, touchedCount, "touched")
    requireDistinct(accumulator, values, "accumulator and values")
    requireDistinct(marks, indices, "marks and indices")
    requireDistinct(marks, touched, "marks and touched")
    validateIndices(indices, indexOffset, count, accumulator.size, "indices")
    var newTouches = 0
    for (k in 0 until count) if (marks[indices[indexOffset + k]] != epoch) newTouches++
    requireWindow(touched.size, touchedOffset + touchedCount, newTouches, "touched capacity")
    requireNonoverlap(
        indices, indexOffset, count, touched, touchedOffset, touchedCount + newTouches, "indices and touched",
    )
}

@Suppress("LongParameterList")
private fun validateGather(
    touched: IntArray,
    touchedOffset: Int,
    touchedCount: Int,
    accumulator: DoubleArray,
    outIndices: IntArray,
    outIndexOffset: Int,
    outValues: DoubleArray,
    outValueOffset: Int,
) {
    requireWindow(touched.size, touchedOffset, touchedCount, "touched")
    requireWindow(outIndices.size, outIndexOffset, touchedCount, "output indices")
    requireWindow(outValues.size, outValueOffset, touchedCount, "output values")
    requireDistinct(accumulator, outValues, "accumulator and output values")
    requireNonoverlap(
        touched, touchedOffset, touchedCount, outIndices, outIndexOffset, touchedCount,
        "touched and output indices",
    )
    validateIndices(touched, touchedOffset, touchedCount, accumulator.size, "touched")
}

private fun validateIndices(indices: IntArray, offset: Int, count: Int, dimension: Int, name: String) {
    for (k in 0 until count) require(indices[offset + k] in 0 until dimension) { "$name entry is out of bounds" }
}

private fun requireWindow(length: Int, offset: Int, count: Int, name: String) {
    require(offset >= 0 && count >= 0 && offset.toLong() + count <= length) { "$name window is out of bounds" }
}

private fun requireDistinct(first: Any, second: Any, name: String) {
    require(first !== second) { "$name must use distinct buffers" }
}

private fun requireNonoverlap(
    first: IntArray,
    firstOffset: Int,
    firstCount: Int,
    second: IntArray,
    secondOffset: Int,
    secondCount: Int,
    name: String,
) {
    require(first !== second || firstOffset + firstCount <= secondOffset || secondOffset + secondCount <= firstOffset) {
        "$name windows must not overlap"
    }
}
