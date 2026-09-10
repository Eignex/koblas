package com.eignex.koblas.sparse.internal

import kotlin.math.abs

/**
 * Portable numerical leaves for compatible sparse accumulation contracts.
 *
 * Product accumulation assigns the first contribution directly; workspace accumulation starts from positive
 * zero. Keeping those entry points separate makes their signed-zero and evaluation-order policies explicit.
 */
internal object SparseAccumulationKernels {
    @Suppress("LongParameterList")
    fun mergeRankOneColumn(
        alpha: Double,
        column: Int,
        x: DoubleArray,
        support: IntArray,
        supportStart: Int,
        supportEnd: Int,
        sourceRows: IntArray,
        sourceValues: DoubleArray,
        sourceStart: Int,
        sourceEnd: Int,
        outRows: IntArray,
        outValues: DoubleArray,
        outOffset: Int,
    ): Int {
        var source = sourceStart
        var update = supportStart
        var written = 0
        while (source < sourceEnd || update < supportEnd) {
            val sourceRow = if (source < sourceEnd) sourceRows[source] else Int.MAX_VALUE
            val updateRow = if (update < supportEnd) support[update] else Int.MAX_VALUE
            when {
                sourceRow < updateRow -> {
                    outRows[outOffset + written] = sourceRow
                    outValues[outOffset + written] = sourceValues[source]
                    source++
                }

                updateRow < sourceRow -> {
                    outRows[outOffset + written] = updateRow
                    outValues[outOffset + written] = 0.0 + (alpha * x[column]) * x[updateRow]
                    update++
                }

                else -> {
                    outRows[outOffset + written] = sourceRow
                    outValues[outOffset + written] = sourceValues[source] + (alpha * x[column]) * x[sourceRow]
                    source++
                    update++
                }
            }
            written++
        }
        return written
    }

    @Suppress("LongParameterList")
    fun mergeRankTwoColumn(
        alpha: Double,
        column: Int,
        x: DoubleArray,
        xSupport: IntArray,
        xStart: Int,
        xEnd: Int,
        y: DoubleArray,
        ySupport: IntArray,
        yStart: Int,
        yEnd: Int,
        sourceRows: IntArray,
        sourceValues: DoubleArray,
        sourceStart: Int,
        sourceEnd: Int,
        outRows: IntArray,
        outValues: DoubleArray,
        outOffset: Int,
    ): Int {
        var source = sourceStart
        var xUpdate = xStart
        var yUpdate = yStart
        var written = 0
        while (source < sourceEnd || xUpdate < xEnd || yUpdate < yEnd) {
            val sourceRow = if (source < sourceEnd) sourceRows[source] else Int.MAX_VALUE
            val xRow = if (xUpdate < xEnd) xSupport[xUpdate] else Int.MAX_VALUE
            val yRow = if (yUpdate < yEnd) ySupport[yUpdate] else Int.MAX_VALUE
            val updateRow = minOf(xRow, yRow)
            if (sourceRow < updateRow) {
                outRows[outOffset + written] = sourceRow
                outValues[outOffset + written] = sourceValues[source]
                source++
            } else {
                var value = if (sourceRow == updateRow) sourceValues[source++] else 0.0
                if (xRow == updateRow) {
                    value += (alpha * y[column]) * x[updateRow]
                    xUpdate++
                }
                if (yRow == updateRow) {
                    value += (alpha * x[column]) * y[updateRow]
                    yUpdate++
                }
                outRows[outOffset + written] = updateRow
                outValues[outOffset + written] = value
            }
            written++
        }
        return written
    }

    @Suppress("LongParameterList")
    fun updateRankOneDenseColumn(
        alpha: Double,
        column: Int,
        x: DoubleArray,
        lower: Boolean,
        rows: Int,
        sourceRows: IntArray,
        sourceValues: DoubleArray,
        sourceStart: Int,
        sourceEnd: Int,
        outRows: IntArray,
        outValues: DoubleArray,
        outOffset: Int,
    ): Int {
        var source = sourceStart
        var written = 0
        val active = x[column] != 0.0
        for (row in 0 until rows) {
            val stored = source < sourceEnd && sourceRows[source] == row
            val selected = if (lower) row >= column else row <= column
            if (selected && active) {
                outRows[outOffset + written] = row
                outValues[outOffset + written] = (if (stored) sourceValues[source] else 0.0) +
                    (alpha * x[column]) * x[row]
                written++
                if (stored) source++
            } else if (stored) {
                outRows[outOffset + written] = row
                outValues[outOffset + written] = sourceValues[source++]
                written++
            }
        }
        return written
    }

    @Suppress("LongParameterList")
    fun updateRankTwoDenseColumn(
        alpha: Double,
        column: Int,
        x: DoubleArray,
        y: DoubleArray,
        lower: Boolean,
        rows: Int,
        sourceRows: IntArray,
        sourceValues: DoubleArray,
        sourceStart: Int,
        sourceEnd: Int,
        outRows: IntArray,
        outValues: DoubleArray,
        outOffset: Int,
    ): Int {
        var source = sourceStart
        var written = 0
        val active = x[column] != 0.0 || y[column] != 0.0
        for (row in 0 until rows) {
            val stored = source < sourceEnd && sourceRows[source] == row
            val selected = if (lower) row >= column else row <= column
            if (selected && active) {
                var value = if (stored) sourceValues[source] else 0.0
                value += (alpha * y[column]) * x[row]
                value += (alpha * x[column]) * y[row]
                outRows[outOffset + written] = row
                outValues[outOffset + written] = value
                written++
                if (stored) source++
            } else if (stored) {
                outRows[outOffset + written] = row
                outValues[outOffset + written] = sourceValues[source++]
                written++
            }
        }
        return written
    }

    @Suppress("LongParameterList")
    fun mergeScaledColumns(
        alpha: Double,
        leftRows: IntArray,
        leftValues: DoubleArray,
        leftStart: Int,
        leftEnd: Int,
        rightRows: IntArray,
        rightValues: DoubleArray,
        rightStart: Int,
        rightEnd: Int,
        outRows: IntArray,
        outValues: DoubleArray,
        outOffset: Int,
    ): Int {
        var left = leftStart
        var right = rightStart
        var written = 0
        while (left < leftEnd || right < rightEnd) {
            val leftRow = if (left < leftEnd) leftRows[left] else Int.MAX_VALUE
            val rightRow = if (right < rightEnd) rightRows[right] else Int.MAX_VALUE
            when {
                leftRow < rightRow -> {
                    outRows[outOffset + written] = leftRow
                    outValues[outOffset + written] = if (alpha == 0.0) alpha else alpha * leftValues[left]
                    left++
                }

                rightRow < leftRow -> {
                    outRows[outOffset + written] = rightRow
                    outValues[outOffset + written] = rightValues[right]
                    right++
                }

                else -> {
                    outRows[outOffset + written] = leftRow
                    outValues[outOffset + written] = if (alpha == 0.0) {
                        rightValues[right]
                    } else {
                        alpha * leftValues[left] + rightValues[right]
                    }
                    left++
                    right++
                }
            }
            written++
        }
        return written
    }

    @Suppress("LongParameterList")
    fun addProductColumnToDense(
        alpha: Double,
        leftPointers: IntArray,
        leftRows: IntArray,
        leftValues: DoubleArray,
        rightRows: IntArray,
        rightValues: DoubleArray,
        rightStart: Int,
        rightEnd: Int,
        firstRow: Int,
        lastRow: Int,
        destination: DoubleArray,
        destinationOffset: Int,
    ) {
        for (rightPosition in rightStart until rightEnd) {
            val leftColumn = rightRows[rightPosition]
            val rightValue = rightValues[rightPosition]
            for (leftPosition in leftPointers[leftColumn] until leftPointers[leftColumn + 1]) {
                val row = leftRows[leftPosition]
                if (row >= firstRow && row < lastRow) {
                    destination[destinationOffset + row] += alpha * (leftValues[leftPosition] * rightValue)
                }
            }
        }
    }

    @Suppress("LongParameterList")
    fun accumulateProductSlice(
        rowIndices: IntArray,
        coefficients: DoubleArray,
        start: Int,
        end: Int,
        factor: Double,
        firstRow: Int,
        lastRow: Int,
        epoch: Int,
        sums: DoubleArray,
        marks: IntArray,
        touched: IntArray,
        touchedCount: Int,
    ): Int {
        var used = touchedCount
        for (position in start until end) {
            val row = rowIndices[position]
            if (row < firstRow || row >= lastRow) continue
            val contribution = coefficients[position] * factor
            if (marks[row] != epoch) {
                marks[row] = epoch
                sums[row] = contribution
                touched[used++] = row
            } else {
                sums[row] += contribution
            }
        }
        return used
    }

    @Suppress("LongParameterList")
    fun accumulateProductPattern(
        rowIndices: IntArray,
        start: Int,
        end: Int,
        firstRow: Int,
        lastRow: Int,
        zero: Double,
        epoch: Int,
        sums: DoubleArray,
        marks: IntArray,
        touched: IntArray,
        touchedCount: Int,
    ): Int {
        var used = touchedCount
        for (position in start until end) {
            val row = rowIndices[position]
            if (row < firstRow || row >= lastRow || marks[row] == epoch) continue
            marks[row] = epoch
            sums[row] = zero
            touched[used++] = row
        }
        return used
    }

    @Suppress("LongParameterList")
    fun accumulateIndirectProductSlice(
        rows: IntArray,
        positions: IntArray,
        coefficients: DoubleArray,
        start: Int,
        end: Int,
        factor: Double,
        firstRow: Int,
        lastRow: Int,
        epoch: Int,
        sums: DoubleArray,
        marks: IntArray,
        touched: IntArray,
        touchedCount: Int,
    ): Int {
        var used = touchedCount
        for (at in start until end) {
            val row = rows[at]
            if (row < firstRow || row >= lastRow) continue
            val contribution = coefficients[positions[at]] * factor
            if (marks[row] != epoch) {
                marks[row] = epoch
                sums[row] = contribution
                touched[used++] = row
            } else {
                sums[row] += contribution
            }
        }
        return used
    }

    @Suppress("LongParameterList")
    fun emitScaledSupport(
        alpha: Double,
        touched: IntArray,
        touchedCount: Int,
        sums: DoubleArray,
        outRows: IntArray,
        outValues: DoubleArray,
        outOffset: Int,
    ) {
        if (alpha == 0.0) {
            for (at in 0 until touchedCount) {
                outRows[outOffset + at] = touched[at]
                outValues[outOffset + at] = alpha
            }
        } else {
            for (at in 0 until touchedCount) {
                val row = touched[at]
                outRows[outOffset + at] = row
                outValues[outOffset + at] = alpha * sums[row]
            }
        }
    }

    fun addScaledSupportToDense(
        alpha: Double,
        touched: IntArray,
        touchedCount: Int,
        sums: DoubleArray,
        destination: DoubleArray,
        destinationOffset: Int,
    ) {
        for (at in 0 until touchedCount) {
            val row = touched[at]
            destination[destinationOffset + row] += alpha * sums[row]
        }
    }

    @Suppress("LongParameterList")
    fun scatterWorkspace(
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
        var total = touchedCount
        for (k in 0 until count) {
            val index = indices[indexOffset + k]
            if (marks[index] != epoch) {
                accumulator[index] = 0.0
                marks[index] = epoch
                touched[touchedOffset + total] = index
                total++
            }
            accumulator[index] += alpha * values[valueOffset + k]
        }
        return total
    }

    @Suppress("LongParameterList")
    fun scatterWorkspaceChecked(
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
        nonfiniteFlag: Int,
        underflowFlag: Int,
    ): Int {
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
            if (!product.isFinite() || !updated.isFinite()) flags = flags or nonfiniteFlag
            if (alpha != 0.0 && value != 0.0 && product == 0.0) flags = flags or underflowFlag
            accumulator[index] = updated
        }
        arithmeticStatus[statusOffset] = flags
        return total
    }

    @Suppress("LongParameterList")
    fun gatherWorkspace(
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        accumulator: DoubleArray,
        outIndices: IntArray,
        outIndexOffset: Int,
        outValues: DoubleArray,
        outValueOffset: Int,
        compactExactZeros: Boolean,
        marks: IntArray?,
    ): Int {
        var written = 0
        for (k in 0 until touchedCount) {
            val index = touched[touchedOffset + k]
            val value = accumulator[index]
            if (!compactExactZeros || value != 0.0) {
                outIndices[outIndexOffset + written] = index
                outValues[outValueOffset + written] = value
                written++
            }
            if (marks != null) {
                accumulator[index] = 0.0
                marks[index] = 0
            }
        }
        return written
    }

    @Suppress("LongParameterList")
    fun activeMaximum(
        rowIndices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        activeRows: BooleanArray,
    ): Double {
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
    fun selectPivotCandidates(
        rowIndices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        activeRows: BooleanArray,
        absoluteTolerance: Double,
        relativeCutoff: Double,
        outPositions: IntArray,
        outOffset: Int,
    ): Int {
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
}
