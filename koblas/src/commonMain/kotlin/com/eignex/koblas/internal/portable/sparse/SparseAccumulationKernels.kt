@file:Suppress("NOTHING_TO_INLINE") // keep run-level extraction from adding a hot call per CSC slice or column

package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.internal.sparse.*

/**
 * Portable numerical leaves for the sparse accumulation contracts the matrix algorithms share.
 *
 * Product accumulation assigns a row's first contribution and adds to it afterwards, which is what the epoch
 * marks are for.
 *
 * Every leaf writes only through the slices it is handed and allocates nothing.
 */
internal object SparseAccumulationKernels {
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
