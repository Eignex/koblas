package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.scalar.sparse.SparseAccumulationKernels

/** Adds the selected triangle of `alpha · op(A) · op(A)ᵀ` directly into dense [c]. */
@Suppress("LongParameterList") // the operands plus the borrowed scratch set
internal fun symmetricRankInto(
    alpha: Double,
    a: SparseMatrix,
    transpose: Boolean,
    c: DenseMatrix,
    lower: Boolean,
    sums: DoubleArray,
    touchedAt: IntArray,
    touched: IntArray,
    rowPointers: IntArray,
    adjacentColumns: IntArray,
    adjacentPositions: IntArray,
    rowCursor: IntArray,
) {
    val order = c.rows
    touchedAt.fill(0, 0, order)
    buildRowAdjacency(a, rowPointers, adjacentColumns, adjacentPositions, rowCursor)
    for (j in 0 until order) {
        val used = accumulateRankColumn(
            a, transpose, j, lower, sums, touchedAt, touched,
            rowPointers, adjacentColumns, adjacentPositions,
        )
        SparseAccumulationKernels.addScaledSupportToDense(
            alpha,
            touched,
            used,
            sums,
            c.values,
            c.offset + j * c.leadingDimension,
        )
    }
}

/** Accumulates one selected rank-update column while adjacency and triangle scheduling stay structural policy. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList")
private fun accumulateRankColumn(
    a: SparseMatrix,
    transpose: Boolean,
    j: Int,
    lower: Boolean,
    sums: DoubleArray,
    touchedAt: IntArray,
    touched: IntArray,
    rowPointers: IntArray,
    adjacentColumns: IntArray,
    adjacentPositions: IntArray,
): Int {
    val order = if (transpose) a.cols else a.rows
    val firstRow = if (lower) j else 0
    val lastRow = if (lower) order else j + 1
    val epoch = j + 1
    var used = 0
    if (!transpose) {
        for (at in rowPointers[j] until rowPointers[j + 1]) {
            val p = adjacentColumns[at]
            val jp = adjacentPositions[at]
            used = SparseAccumulationKernels.accumulateProductSlice(
                a.rowIndices, a.values, a.colPointers[p], a.colPointers[p + 1], a.values[jp], firstRow, lastRow,
                epoch, sums, touchedAt, touched, used,
            )
        }
        return used
    }
    for (jp in a.colPointers[j] until a.colPointers[j + 1]) {
        val p = a.rowIndices[jp]
        used = SparseAccumulationKernels.accumulateIndirectProductSlice(
            adjacentColumns, adjacentPositions, a.values, rowPointers[p], rowPointers[p + 1], a.values[jp],
            firstRow, lastRow, epoch, sums, touchedAt, touched, used,
        )
    }
    return used
}

/** Builds row-to-stored-entry adjacency once, in ascending source-column order. */
@OptIn(UnsafeKoblasApi::class)
private fun buildRowAdjacency(
    a: SparseMatrix,
    rowPointers: IntArray,
    adjacentColumns: IntArray,
    adjacentPositions: IntArray,
    rowCursor: IntArray,
) {
    rowPointers.fill(0, 0, a.rows + 1)
    for (position in 0 until a.nnz) rowPointers[a.rowIndices[position] + 1]++
    for (row in 0 until a.rows) rowPointers[row + 1] += rowPointers[row]
    for (row in 0 until a.rows) rowCursor[row] = rowPointers[row]
    for (column in 0 until a.cols) {
        for (position in a.colPointers[column] until a.colPointers[column + 1]) {
            val row = a.rowIndices[position]
            val target = rowCursor[row]++
            adjacentColumns[target] = column
            adjacentPositions[target] = position
        }
    }
}
