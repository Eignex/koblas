package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix

/**
 * `A · B` for two CSC operands, by Gustavson's method: one column of the result at a time, accumulated in a
 * dense scratch row indexed by the result's rows and read back through the positions it touched.
 *
 * The scratch is what makes this linear in the work rather than in the shape. A column of `B` selects the
 * columns of `A` that contribute to it, and each contribution scatters into the scratch; the positions
 * touched are collected as they are first written, so the column is read back without sweeping the rows that
 * stayed empty.
 *
 * Which positions those are is structural: an operand's stored zero selects and scatters like any other
 * entry, so the result stores what the two patterns meet at, as an entry the arithmetic cancels to zero is
 * also kept.
 */
internal fun multiplySparse(
    a: SparseMatrix,
    b: SparseMatrix,
    alpha: Double = 1.0,
    lower: Boolean? = null,
): SparseMatrix {
    val rows = a.rows
    val values = DoubleArray(rows)
    // Positive column epochs distinguish first touches without clearing between columns.
    val touchedIn = IntArray(rows)
    val touched = IntArray(rows)

    val colPtr = IntArray(b.cols + 1)
    var outIdx = IntArray(a.nnz + b.nnz)
    var outVal = DoubleArray(outIdx.size)
    var count = 0

    for (j in 0 until b.cols) {
        var used = 0
        val epoch = j + 1
        val firstRow = if (lower == true) j else 0
        val lastRow = if (lower == false) j + 1 else rows
        for (bp in b.colPtr[j] until b.colPtr[j + 1]) {
            val l = b.rowIdx[bp]
            // A stored zero of B contributes its column of A as stored zeros rather than dropping it: the
            // pattern of the product is the pattern of the operands, whatever the arithmetic makes of it.
            used = if (alpha == 0.0) {
                SparseAccumulationKernels.accumulateProductPattern(
                    a.rowIdx, a.colPtr[l], a.colPtr[l + 1], firstRow, lastRow,
                    alpha, epoch, values, touchedIn, touched, used,
                )
            } else {
                SparseAccumulationKernels.accumulateProductSlice(
                    a.rowIdx, a.values, a.colPtr[l], a.colPtr[l + 1], b.values[bp], firstRow, lastRow,
                    epoch, values, touchedIn, touched, used,
                )
            }
        }
        if (count + used > outIdx.size) {
            val grown = maxOf(outIdx.size * 2, count + used)
            outIdx = outIdx.copyOf(grown)
            outVal = outVal.copyOf(grown)
        }
        // Rows arrive in whatever order the contributing columns held them, and CSC wants them ascending.
        // Sorted where they were collected, since the scratch is already this column's and nothing else
        // reads it before the next column overwrites the same prefix.
        touched.sort(0, used)
        SparseAccumulationKernels.emitScaledSupport(alpha, touched, used, values, outIdx, outVal, count)
        count += used
        colPtr[j + 1] = count
    }
    // Each column's rows were sorted where they were collected, and a scatter list holds each row once.
    return SparseMatrix.wrapTrusted(rows, b.cols, colPtr, outIdx.copyOf(count), outVal.copyOf(count))
}

/**
 * Adds `alpha · A · B` directly into dense [c]. Both sparse operands are already in the requested
 * orientation. [lower] limits writes to one triangle when non-null. Stored traversal deliberately avoids
 * products with implicit sparse zeros.
 */
internal fun multiplySparseInto(
    alpha: Double,
    a: SparseMatrix,
    b: SparseMatrix,
    c: DenseMatrix,
    lower: Boolean? = null,
) {
    if (alpha == 0.0) return
    for (j in 0 until b.cols) {
        SparseAccumulationKernels.addProductColumnToDense(
            alpha, a.colPtr, a.rowIdx, a.values, b.rowIdx, b.values, b.colPtr[j], b.colPtr[j + 1],
            if (lower == true) j else 0,
            if (lower == false) j + 1 else a.rows,
            c.data, j * c.rows,
        )
    }
}

/** Fresh selected triangle of `op(A) · op(A)ᵀ` without materializing a transpose or full product. */
internal fun symmetricRankProduct(a: SparseMatrix, transpose: Boolean, lower: Boolean): SparseMatrix {
    val order = if (transpose) a.cols else a.rows
    val sums = DoubleArray(order)
    val touchedAt = IntArray(order)
    val touched = IntArray(order)
    val rowPointers = IntArray(a.rows + 1)
    val adjacentColumns = IntArray(a.nnz)
    val adjacentPositions = IntArray(a.nnz)
    buildRowAdjacency(a, rowPointers, adjacentColumns, adjacentPositions, IntArray(a.rows))
    val pointers = IntArray(order + 1)
    var rows = IntArray(maxOf(1, a.nnz))
    var values = DoubleArray(rows.size)
    var count = 0
    for (j in 0 until order) {
        val used = accumulateRankColumn(
            a, transpose, j, lower, sums, touchedAt, touched,
            rowPointers, adjacentColumns, adjacentPositions,
        )
        if (count + used > rows.size) {
            val size = maxOf(rows.size * 2, count + used)
            rows = rows.copyOf(size)
            values = values.copyOf(size)
        }
        touched.sort(0, used)
        SparseAccumulationKernels.emitScaledSupport(1.0, touched, used, sums, rows, values, count)
        count += used
        pointers[j + 1] = count
    }
    return SparseMatrix.wrapTrusted(order, order, pointers, rows.copyOf(count), values.copyOf(count))
}

/** Adds the selected triangle of `alpha · op(A) · op(A)ᵀ` directly into dense [c]. */
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
    touchedAt.fill(0)
    buildRowAdjacency(a, rowPointers, adjacentColumns, adjacentPositions, rowCursor)
    for (j in 0 until order) {
        val used = accumulateRankColumn(
            a, transpose, j, lower, sums, touchedAt, touched,
            rowPointers, adjacentColumns, adjacentPositions,
        )
        SparseAccumulationKernels.addScaledSupportToDense(alpha, touched, used, sums, c.data, j * order)
    }
}

/** Accumulates one selected SYRK column while adjacency and triangle scheduling stay structural policy. */
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
    val firstRow = if (lower) j else 0
    val lastRow = if (lower) sums.size else j + 1
    val epoch = j + 1
    var used = 0
    if (!transpose) {
        for (at in rowPointers[j] until rowPointers[j + 1]) {
            val p = adjacentColumns[at]
            val jp = adjacentPositions[at]
            used = SparseAccumulationKernels.accumulateProductSlice(
                a.rowIdx, a.values, a.colPtr[p], a.colPtr[p + 1], a.values[jp], firstRow, lastRow,
                epoch, sums, touchedAt, touched, used,
            )
        }
        return used
    }
    for (jp in a.colPtr[j] until a.colPtr[j + 1]) {
        val p = a.rowIdx[jp]
        used = SparseAccumulationKernels.accumulateIndirectProductSlice(
            adjacentColumns, adjacentPositions, a.values, rowPointers[p], rowPointers[p + 1], a.values[jp],
            firstRow, lastRow, epoch, sums, touchedAt, touched, used,
        )
    }
    return used
}

/** Builds row-to-stored-entry adjacency once, in ascending source-column order. */
private fun buildRowAdjacency(
    a: SparseMatrix,
    rowPointers: IntArray,
    adjacentColumns: IntArray,
    adjacentPositions: IntArray,
    rowCursor: IntArray,
) {
    rowPointers.fill(0)
    for (position in a.rowIdx.indices) rowPointers[a.rowIdx[position] + 1]++
    for (row in 0 until a.rows) rowPointers[row + 1] += rowPointers[row]
    for (row in 0 until a.rows) rowCursor[row] = rowPointers[row]
    for (column in 0 until a.cols) {
        for (position in a.colPtr[column] until a.colPtr[column + 1]) {
            val row = a.rowIdx[position]
            val target = rowCursor[row]++
            adjacentColumns[target] = column
            adjacentPositions[target] = position
        }
    }
}
