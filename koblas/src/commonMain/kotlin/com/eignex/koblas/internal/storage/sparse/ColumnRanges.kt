package com.eignex.koblas.internal.storage.sparse

/**
 * Where the selected part of a CSC column starts, with the diagonal counted as selected.
 *
 * A column's rows ascend, so what a triangle selects from it is a prefix or a suffix rather than a scattered
 * subset. That is what lets a selected half be handed to a panel leaf as one run, and these four functions
 * are what decide which entries a panel is actually given.
 */
internal fun selectedRunStart(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) lowerBound(rowIndices, from, to, column) else from

/** Where the selected part of a column ends, the counterpart of [selectedRunStart]. */
internal fun selectedRunEnd(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) to else lowerBound(rowIndices, from, to, column + 1)

/**
 * Where the strictly triangular part of a column starts, which is the selected part without its diagonal.
 *
 * Without rather than minus: an ascending run has its diagonal at the inner end of the selected part if it
 * is stored at all, so dropping it is a bound one position further in and never a hole in the middle.
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
