@file:Suppress("NOTHING_TO_INLINE") // Avoid a hot call per CSC slice or column.

package com.eignex.koblas.internal.kernels.scalar.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.forEachStored
import com.eignex.koblas.internal.storage.sparse.SparseRowSums
import kotlin.math.abs

/**
 * Portable numerical leaves for the sparse accumulation contracts the matrix algorithms share.
 *
 * Product accumulation assigns a row's first contribution and adds to it afterwards, which is what the epoch
 * marks are for.
 *
 * Slice leaves write only through the buffers they are handed. Row reductions own support-sized storage.
 */
@Suppress("LongParameterList")
internal inline fun accumulateProductSlice(
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
    coefficientIndex: (Int) -> Int = { it },
): Int {
    var used = touchedCount
    for (position in start until end) {
        val row = rowIndices[position]
        if (row < firstRow || row >= lastRow) continue
        val contribution = coefficients[coefficientIndex(position)] * factor
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

internal inline fun addScaledSupportToDense(
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

/** Only stored rows need accumulation, even when the logical row domain is too large for a dense buffer. */
internal fun sparseMatrixInfinityNorm(a: SparseMatrix): Double {
    if (a.nnz == 0) return 0.0
    val sums = SparseRowSums()
    a.forEachStored { i, _, value -> sums.add(i, abs(value)) }
    return sums.maximumOrZero()
}
