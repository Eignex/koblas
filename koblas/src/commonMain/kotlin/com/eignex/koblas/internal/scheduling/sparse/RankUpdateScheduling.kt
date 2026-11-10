package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.scalar.sparse.*
import com.eignex.koblas.internal.storage.borrowI32
import com.eignex.koblas.internal.storage.sparse.pointerLength

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
        addScaledSupportToDense(alpha, touched, used, sums, c.values, c.offset + j * c.leadingDimension)
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
            used = accumulateProductSlice(
                a.rowIndices, a.values, a.colPointers[p], a.colPointers[p + 1], a.values[jp], firstRow, lastRow,
                epoch, sums, touchedAt, touched, used,
            )
        }
        return used
    }
    for (jp in a.colPointers[j] until a.colPointers[j + 1]) {
        val p = a.rowIndices[jp]
        used = accumulateProductSlice(
            adjacentColumns, a.values, rowPointers[p], rowPointers[p + 1], a.values[jp],
            firstRow, lastRow, epoch, sums, touchedAt, touched, used,
        ) { adjacentPositions[it] }
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

/**
 * Borrows the complete rank-update scratch set as one exception-safe inline scope. No holder object is
 * created; each buffer remains an independent arrays loan and is returned if a later loan or the kernel
 * throws.
 */
internal inline fun <T> withSymmetricRankScratch(
    arrays: Scratch?,
    order: Int,
    sourceRows: Int,
    sourceEntries: Int,
    crossinline block: (
        sums: DoubleArray,
        touchedAt: IntArray,
        touched: IntArray,
        rowPointers: IntArray,
        adjacentColumns: IntArray,
        adjacentPositions: IntArray,
        rowCursor: IntArray,
    ) -> T,
): T = arrays.borrow(order) { sums ->
    arrays.borrowI32(order) { touchedAt ->
        arrays.borrowI32(order) { touched ->
            arrays.borrowI32(pointerLength(sourceRows, "syrk")) { rowPointers ->
                arrays.borrowI32(sourceEntries) { adjacentColumns ->
                    arrays.borrowI32(sourceEntries) { adjacentPositions ->
                        arrays.borrowI32(sourceRows) { rowCursor ->
                            block(sums, touchedAt, touched, rowPointers, adjacentColumns, adjacentPositions, rowCursor)
                        }
                    }
                }
            }
        }
    }
}
