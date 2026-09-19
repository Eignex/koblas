package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.requireShape

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
 * entry, so the result stores what the two patterns meet at, and an entry the arithmetic cancels to zero is
 * kept as well.
 */
@OptIn(UnsafeKoblasApi::class)
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

    val outPointers = IntArray(b.cols + 1)
    val builder = SupportBuilder(a.nnz.toLong() + b.nnz)

    for (j in 0 until b.cols) {
        var used = 0
        val epoch = j + 1
        val firstRow = if (lower == true) j else 0
        val lastRow = if (lower == false) j + 1 else rows
        for (bp in b.colPointers[j] until b.colPointers[j + 1]) {
            val l = b.rowIndices[bp]
            // A stored zero of B contributes its column of A as stored zeros rather than dropping it: the
            // pattern of the product is the pattern of the operands, whatever the arithmetic makes of it.
            used = if (alpha == 0.0) {
                SparseAccumulationKernels.accumulateProductPattern(
                    a.rowIndices, a.colPointers[l], a.colPointers[l + 1], firstRow, lastRow,
                    alpha, epoch, values, touchedIn, touched, used,
                )
            } else {
                SparseAccumulationKernels.accumulateProductSlice(
                    a.rowIndices, a.values, a.colPointers[l], a.colPointers[l + 1], b.values[bp],
                    firstRow, lastRow, epoch, values, touchedIn, touched, used,
                )
            }
        }
        builder.reserve(used)
        // Rows arrive in whatever order the contributing columns held them, and CSC wants them ascending.
        // Sorted where they were collected, since the scratch is already this column's and nothing else
        // reads it before the next column overwrites the same prefix.
        touched.sort(0, used)
        SparseAccumulationKernels.emitScaledSupport(
            alpha,
            touched,
            used,
            values,
            builder.rows,
            builder.values,
            builder.size,
        )
        builder.advance(used)
        outPointers[j + 1] = builder.size
    }
    // Each column's rows were sorted where they were collected, and a scatter list holds each row once.
    return builder.finish(rows, b.cols, outPointers)
}

/**
 * Adds `alpha · A · B` directly into dense [c]. Both sparse operands are already in the requested
 * orientation. [lower] limits writes to one triangle when non-null. Stored traversal deliberately avoids
 * products with implicit sparse zeros.
 */
@OptIn(UnsafeKoblasApi::class)
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
            alpha, a.colPointers, a.rowIndices, a.values, b.rowIndices, b.values,
            b.colPointers[j], b.colPointers[j + 1],
            if (lower == true) j else 0,
            if (lower == false) j + 1 else a.rows,
            c.values, j * c.rows,
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
    val outPointers = IntArray(order + 1)
    val builder = SupportBuilder(maxOf(1, a.nnz).toLong())
    for (j in 0 until order) {
        val used = accumulateRankColumn(
            a, transpose, j, lower, sums, touchedAt, touched,
            rowPointers, adjacentColumns, adjacentPositions,
        )
        builder.reserve(used)
        touched.sort(0, used)
        SparseAccumulationKernels.emitScaledSupport(
            1.0,
            touched,
            used,
            sums,
            builder.rows,
            builder.values,
            builder.size,
        )
        builder.advance(used)
        outPointers[j + 1] = builder.size
    }
    return builder.finish(order, order, outPointers)
}

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
        SparseAccumulationKernels.addScaledSupportToDense(alpha, touched, used, sums, c.values, j * order)
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

/**
 * The growing support of a discovered CSC result.
 *
 * Capacity arithmetic is in `Long` because the two operands' entry counts multiply out to more than an array
 * can index long before they overflow individually, and a product that cannot be stored has to say so rather
 * than wrap into a negative length.
 */
private class SupportBuilder(initialCapacity: Long) {
    var rows: IntArray = IntArray(initialCapacity.coerceIn(0L, MAX_SUPPORT).toInt())
        private set
    var values: DoubleArray = DoubleArray(rows.size)
        private set
    var size: Int = 0
        private set

    fun reserve(additional: Int) {
        val required = size.toLong() + additional
        if (required <= rows.size) return
        requireShape(
            required <= MAX_SUPPORT,
        ) { "sparse product needs $required stored entries, more than one array can hold" }
        val grown = maxOf(rows.size.toLong() * 2, required).coerceAtMost(MAX_SUPPORT).toInt()
        rows = rows.copyOf(grown)
        values = values.copyOf(grown)
    }

    fun advance(written: Int) {
        size += written
    }

    fun finish(rows: Int, cols: Int, pointers: IntArray): SparseMatrix =
        SparseMatrix.wrapTrusted(rows, cols, pointers, this.rows.copyOf(size), values.copyOf(size))

    private companion object {
        const val MAX_SUPPORT = Int.MAX_VALUE.toLong()
    }
}
