@file:Suppress("NOTHING_TO_INLINE") // keep run-level extraction from adding a hot call per CSC slice or column

package com.eignex.koblas.sparse.internal

/**
 * Portable numerical leaves for the sparse accumulation contracts the matrix algorithms share.
 *
 * Product accumulation assigns its first contribution directly rather than adding to a zero, so a product
 * whose only term is a negative zero keeps its sign. The workspace accumulation
 * [com.eignex.koblas.sparse.SparsePrimitives] publishes starts from positive zero instead, which is why the
 * two are separate entry points rather than one with a flag.
 *
 * Every leaf writes only through the slices it is handed and allocates nothing.
 */
internal object SparseAccumulationKernels {
    /*
     * The two dense-column rank updates write every selected position rather than skipping a column whose
     * update coefficient is zero. Netlib's dsyr and dsyr2 skip it; the dense implementation here does not,
     * because the product it would skip is `0 * infinity` and that is a NaN the caller asked for. These are
     * reached only when an operand is non-finite, and agreeing with the dense routine over the same values is
     * the whole reason that path exists.
     */
    @Suppress("LongParameterList")
    inline fun mergeRankOneColumn(
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
    inline fun mergeRankTwoColumn(
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
    inline fun updateRankOneDenseColumn(
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
        for (row in 0 until rows) {
            val stored = source < sourceEnd && sourceRows[source] == row
            val selected = if (lower) row >= column else row <= column
            if (selected) {
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
    inline fun updateRankTwoDenseColumn(
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
        for (row in 0 until rows) {
            val stored = source < sourceEnd && sourceRows[source] == row
            val selected = if (lower) row >= column else row <= column
            if (selected) {
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
    inline fun mergeScaledColumns(
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
    inline fun addProductColumnToDense(
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
    inline fun accumulateProductSlice(
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
    inline fun accumulateProductPattern(
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
    inline fun accumulateIndirectProductSlice(
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

    /**
     * The rows this column touched, in ascending order, read back from the marks rather than sorted.
     *
     * The scatter collects rows in whatever order the contributing columns held them, and CSC wants them
     * ascending. Two ways to get there: sort what was collected, which costs with the number of entries, or
     * sweep the range the column could have reached, which costs with that range whatever the column holds.
     * Which is cheaper depends on how much of the range was touched, and the caller decides with
     * [sweepsTouchedRows].
     */
    @Suppress("LongParameterList")
    inline fun collectTouchedRows(marks: IntArray, epoch: Int, firstRow: Int, lastRow: Int, touched: IntArray): Int {
        var used = 0
        for (row in firstRow until lastRow) {
            if (marks[row] == epoch) touched[used++] = row
        }
        return used
    }

    @Suppress("LongParameterList")
    inline fun emitScaledSupport(
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

    inline fun addScaledSupportToDense(
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
}
