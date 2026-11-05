package com.eignex.koblas.internal.kernels.scalar.sparse

import com.eignex.koblas.internal.storage.sparse.selectedRunEnd
import com.eignex.koblas.internal.storage.sparse.selectedRunStart

/** A stored symmetric column updates both halves, counting the diagonal once. */
@Suppress("LongParameterList") // one CSC column slice, its triangle flag, and the two vectors
internal fun sparseSymmetricVectorColumn(
    alpha: Double,
    column: Int,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    x: DoubleArray,
    y: DoubleArray,
    lower: Boolean,
) {
    for (position in fromIndex until toIndex) {
        val row = rowIndices[position]
        if (if (lower) row >= column else row <= column) {
            val value = values[position]
            y[row] += alpha * (value * x[column])
            if (row != column) y[column] += alpha * (value * x[row])
        }
    }
}

/**
 * Single-right-hand-side product arithmetic without panel-call, fill and grouping overhead.
 *
 * Products visit the same entries in traversal order. Grouped panels may reassociate reductions,
 * so agreement follows the allowed numerical tolerance rather than requiring identical bits.
 */
@Suppress("LongParameterList") // the column slice, the dense window, and the single output
internal fun sparseGatherProductColumn(
    alpha: Double,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    b: DoubleArray,
    offset: Int,
    indexStride: Int,
    c: DoubleArray,
    cOffset: Int,
) {
    if (fromIndex == toIndex) return
    var sum = 0.0
    for (position in fromIndex until toIndex) {
        sum += values[position] * b[offset + rowIndices[position] * indexStride]
    }
    c[cOffset] += alpha * sum
}

/**
 * `sparseScatterProductPanel` with one right-hand side, for the reason [sparseGatherProductColumn] gives.
 */
@Suppress("LongParameterList") // the column slice, the dense window, and the single output
internal fun sparseScatterProductColumn(
    alpha: Double,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    b: DoubleArray,
    bOffset: Int,
    c: DoubleArray,
    offset: Int,
    indexStride: Int,
) {
    val coefficient = alpha * b[bOffset]
    for (position in fromIndex until toIndex) {
        c[offset + rowIndices[position] * indexStride] += values[position] * coefficient
    }
}

/** `sparseSymmetricLeftColumn` with one right-hand side, written out for the same reason. */
@Suppress("LongParameterList") // one CSC column slice, its triangle flag, and both dense windows
internal fun sparseSymmetricLeftColumnSingle(
    alpha: Double,
    column: Int,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    b: DoubleArray,
    c: DoubleArray,
    offset: Int,
    indexStride: Int,
    lower: Boolean,
    cOffset: Int,
) {
    val start = selectedRunStart(rowIndices, fromIndex, toIndex, column, lower)
    val end = selectedRunEnd(rowIndices, fromIndex, toIndex, column, lower)
    if (start >= end) return
    val pivot = offset + column * indexStride
    val coefficient = b[pivot]
    // Read each stored entry once for both halves; the diagonal contributes only once.
    for (position in start until end) {
        val row = rowIndices[position]
        val t = alpha * values[position]
        val at = offset + row * indexStride
        c[cOffset + row * indexStride] += t * coefficient
        if (row != column) c[cOffset + column * indexStride] += t * b[at]
    }
}
