package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.Scratch
import com.eignex.koblas.borrow
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.sparse.*
import com.eignex.koblas.internal.storage.borrowI32
import com.eignex.koblas.internal.storage.sparse.pointerLength

/** Adds the selected triangle of `alpha · op(A) · op(A)ᵀ` directly into dense [cValues]. */
@Suppress("LongParameterList") // the operands plus the borrowed scratch set
internal fun symmetricRankInto(
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aRows: Int,
    aColumns: Int,
    transpose: Boolean,
    cValues: DoubleArray,
    cOffset: Int,
    cRows: Int,
    cLeadingDimension: Int,
    lower: Boolean,
    sums: DoubleArray,
    touchedAt: IntArray,
    touched: IntArray,
    rowPointers: IntArray,
    adjacentColumns: IntArray,
    adjacentPositions: IntArray,
    rowCursor: IntArray,
) {
    val order = cRows
    touchedAt.fill(0, 0, order)
    buildRowAdjacency(
        aValues,
        aPointers,
        aIndices,
        aRows,
        aColumns,
        rowPointers,
        adjacentColumns,
        adjacentPositions,
        rowCursor,
    )
    for (j in 0 until order) {
        val used = accumulateRankColumn(
            aValues,
            aPointers,
            aIndices,
            aRows,
            aColumns,
            transpose,
            j,
            lower,
            sums,
            touchedAt,
            touched,
            rowPointers,
            adjacentColumns,
            adjacentPositions,
        )
        addScaledSupportToDense(alpha, touched, used, sums, cValues, cOffset + j * cLeadingDimension)
    }
}

/** Accumulates one selected rank-update column while adjacency and triangle scheduling stay structural policy. */
@Suppress("LongParameterList")
private fun accumulateRankColumn(
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aRows: Int,
    aColumns: Int,
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
    val order = if (transpose) aColumns else aRows
    val firstRow = if (lower) j else 0
    val lastRow = if (lower) order else j + 1
    val epoch = j + 1
    var used = 0
    if (!transpose) {
        for (at in rowPointers[j] until rowPointers[j + 1]) {
            val p = adjacentColumns[at]
            val jp = adjacentPositions[at]
            used = accumulateProductSlice(
                aIndices, aValues, aPointers[p], aPointers[p + 1], aValues[jp], firstRow, lastRow,
                epoch, sums, touchedAt, touched, used,
            )
        }
        return used
    }
    for (jp in aPointers[j] until aPointers[j + 1]) {
        val p = aIndices[jp]
        used = accumulateProductSlice(
            adjacentColumns, aValues, rowPointers[p], rowPointers[p + 1], aValues[jp],
            firstRow, lastRow, epoch, sums, touchedAt, touched, used,
        ) { adjacentPositions[it] }
    }
    return used
}

/** Builds row-to-stored-entry adjacency once, in ascending source-column order. */
private fun buildRowAdjacency(
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aRows: Int,
    aColumns: Int,
    rowPointers: IntArray,
    adjacentColumns: IntArray,
    adjacentPositions: IntArray,
    rowCursor: IntArray,
) {
    rowPointers.fill(0, 0, aRows + 1)
    for (position in 0 until aValues.size) rowPointers[aIndices[position] + 1]++
    for (row in 0 until aRows) rowPointers[row + 1] += rowPointers[row]
    for (row in 0 until aRows) rowCursor[row] = rowPointers[row]
    for (column in 0 until aColumns) {
        for (position in aPointers[column] until aPointers[column + 1]) {
            val row = aIndices[position]
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
