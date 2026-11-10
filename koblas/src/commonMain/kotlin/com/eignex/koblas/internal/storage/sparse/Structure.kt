package com.eignex.koblas.internal.storage.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.requireShape

/** Replacing a column preserves every other column and adjusts the following CSC boundaries. */
internal fun SparseMatrix.replaceColumn(column: Int, indices: IntArray, coefficients: DoubleArray) {
    val start = colPointers[column]
    val end = colPointers[column + 1]
    spliceColumn(column, start, end, indices.size) { rows, values ->
        indices.copyInto(rows, start)
        coefficients.copyInto(values, start)
    }
}

/** Structural writes replace buffers together so retained column views follow the new CSC boundaries. */
internal inline fun SparseMatrix.spliceColumn(
    column: Int,
    start: Int,
    end: Int,
    count: Int,
    write: (IntArray, DoubleArray) -> Unit,
) {
    spliceSparseEntries(rowIndices, values, start, end, count) { rows, values ->
        write(rows, values)
        val pointers = colPointers.copyOf()
        val delta = count - (end - start)
        for (j in column + 1..cols) pointers[j] += delta
        replaceStorage(pointers, rows, values)
    }
}

/** Copies the entries outside [start] until [end], leaving [count] replacement slots for [write]. */
internal inline fun spliceSparseEntries(
    indices: IntArray,
    values: DoubleArray,
    start: Int,
    end: Int,
    count: Int,
    write: (IntArray, DoubleArray) -> Unit,
) {
    val size = values.size.toLong() - (end - start) + count
    requireShape(size <= Int.MAX_VALUE) { "stored entry count exceeds Int capacity" }
    val newIndices = IntArray(size.toInt())
    val newValues = DoubleArray(size.toInt())
    indices.copyInto(newIndices, 0, 0, start)
    values.copyInto(newValues, 0, 0, start)
    indices.copyInto(newIndices, start + count, end)
    values.copyInto(newValues, start + count, end)
    write(newIndices, newValues)
}

/**
 * Start of the selected CSC column run, including its diagonal. Ascending rows make a selected triangle
 * a prefix or suffix, so panel leaves receive a contiguous run.
 */
internal fun selectedRunStart(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) lowerBound(rowIndices, from, to, column) else from

/** Where the selected part of a column ends, the counterpart of [selectedRunStart]. */
internal fun selectedRunEnd(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) to else lowerBound(rowIndices, from, to, column + 1)

/**
 * Start of the strictly triangular run. Its diagonal, when stored, is an endpoint, so excluding it never
 * leaves a hole in the run.
 */
internal fun strictRunStart(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) lowerBound(rowIndices, from, to, column + 1) else from

/** Where the strictly triangular part of a column ends, the counterpart of [strictRunStart]. */
internal fun strictRunEnd(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) to else lowerBound(rowIndices, from, to, column)

/**
 * The first position in the ascending run `[from, to)` whose row index is at least [row].
 */
internal fun lowerBound(rowIndices: IntArray, from: Int, to: Int, row: Int): Int {
    // Triangle-only storage puts each boundary at an endpoint, avoiding binary search.
    if (from >= to || rowIndices[from] >= row) return from
    if (rowIndices[to - 1] < row) return to
    var low = from + 1
    var high = to - 1
    while (low < high) {
        val middle = (low + high) ushr 1
        if (rowIndices[middle] < row) low = middle + 1 else high = middle
    }
    return low
}

/**
 * Checks a result's pointer count before allocation. A valid [SparseMatrix] bounds its columns through
 * its pointer array but may have arbitrarily many rows; transposing those rows into columns must report
 * an unrepresentable array length as a shape error rather than an allocation failure.
 */
internal fun pointerLength(columns: Int, what: String): Int {
    requireShape(columns < Int.MAX_VALUE) {
        "$what: a CSC result with $columns columns needs ${columns.toLong() + 1} pointers, more than one array can hold"
    }
    return columns + 1
}

/**
 * Checks extent-indexed scratch before allocation. Empty rank-update sources need no row scratch;
 * callers that do need it reject unrepresentable lengths here.
 */
internal fun scratchLength(extent: Int, what: String): Int {
    requireShape(extent < Int.MAX_VALUE) {
        "$what: scratch for $extent rows needs ${extent.toLong() + 1} entries, more than one array can hold"
    }
    return extent + 1
}

/**
 * CSC transpose, also CSC-to-CSR conversion, preserving stored zeros. Visiting source columns in order
 * constructs ascending, unique output rows, so the result needs no structural revalidation.
 * With [readValues] false, copies only the pattern and leaves values zero: zero multipliers can orient
 * an operand without reading coefficients or allocating a zeroed source copy.
 */
@OptIn(UnsafeKoblasApi::class)
internal fun transposeCsc(a: SparseMatrix, readValues: Boolean = true): SparseMatrix {
    val rows = a.rows
    val cols = a.cols
    val colPointers = a.colPointers
    val rowIndices = a.rowIndices
    val values = a.values
    // The transpose has one column per source row, so this is where an unrepresentable row count is refused.
    val outPointers = IntArray(pointerLength(rows, "transpose"))
    for (k in rowIndices.indices) outPointers[rowIndices[k] + 1]++
    for (i in 0 until rows) outPointers[i + 1] += outPointers[i]
    val outIndices = IntArray(values.size)
    val outValues = DoubleArray(values.size)
    // Each start becomes its column's end as entries arrive. Shifting those ends back to starts afterwards
    // lets the result's own pointers carry the cursors instead of allocating another array of row count.
    for (j in 0 until cols) {
        for (k in colPointers[j] until colPointers[j + 1]) {
            val slot = outPointers[rowIndices[k]]++
            outIndices[slot] = j
            if (readValues) outValues[slot] = values[k]
        }
    }
    for (i in rows downTo 1) outPointers[i] = outPointers[i - 1]
    outPointers[0] = 0
    return SparseMatrix.wrapTrusted(cols, rows, outPointers, outIndices, outValues)
}
