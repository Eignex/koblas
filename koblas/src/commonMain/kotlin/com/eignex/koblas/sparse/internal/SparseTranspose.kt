package com.eignex.koblas.sparse.internal

import com.eignex.koblas.SparseMatrix

/**
 * The CSC transpose, which is also the CSC-to-CSR conversion. Explicitly stored zeros survive, since the
 * transpose is structural.
 *
 * Its own function rather than a method on the portable backend because the portable factorizations
 * transpose too, and reaching the seam for it would route the definition of a routine through whichever
 * backend happens to be registered.
 */
internal fun transposeCsc(a: SparseMatrix): SparseMatrix =
    transposeOf(a.rows, a.cols, a.colPtr, a.rowIdx, a.values, trusted = true)

/**
 * The same over loose arrays, which is what a native binding has: a library hands back column pointers and
 * row indices rather than a validated matrix, and its rows need not ascend within a column. Transposing
 * sorts them, so this is also how such a matrix is admitted at all.
 */
internal fun transposeRaw(
    rows: Int,
    cols: Int,
    colPtr: IntArray,
    rowIdx: IntArray,
    values: DoubleArray,
): SparseMatrix = transposeOf(rows, cols, colPtr, rowIdx, values, trusted = false)

/**
 * The transpose itself. [trusted] says the input already holds the CSC invariant, which makes the output
 * hold it too: this walks the columns in order, so each output column collects its entries by ascending
 * source column, and an input with no repeated coordinate yields no repeated row. An input that has not been
 * checked is exactly the one whose transpose has to be, so a raw transpose validates its result.
 */
@Suppress("LongParameterList") // the CSC triple plus its extents
private fun transposeOf(
    rows: Int,
    cols: Int,
    colPtr: IntArray,
    rowIdx: IntArray,
    values: DoubleArray,
    trusted: Boolean,
): SparseMatrix {
    val outPtr = IntArray(rows + 1)
    for (k in rowIdx.indices) outPtr[rowIdx[k] + 1]++
    for (i in 0 until rows) outPtr[i + 1] += outPtr[i]
    val outIdx = IntArray(values.size)
    val outVal = DoubleArray(values.size)
    val next = outPtr.copyOf()
    for (j in 0 until cols) {
        for (k in colPtr[j] until colPtr[j + 1]) {
            val slot = next[rowIdx[k]]++
            outIdx[slot] = j
            outVal[slot] = values[k]
        }
    }
    return if (trusted) {
        SparseMatrix.wrapTrusted(cols, rows, outPtr, outIdx, outVal)
    } else {
        SparseMatrix.wrap(cols, rows, outPtr, outIdx, outVal)
    }
}

/**
 * A validated CSC matrix over the loose arrays a native binding gets back, whose rows the library need not
 * have sorted within a column.
 *
 * Transposing sorts them, so transposing twice both sorts the pattern and returns it to the orientation it
 * came in. [transposed] keeps the single transpose instead, which is what a library handing back a row form
 * wants: that form is already the transpose in CSC, so one pass both converts and sorts.
 */
internal fun sortedCsc(
    rows: Int,
    cols: Int,
    colPtr: IntArray,
    rowIdx: IntArray,
    values: DoubleArray,
    transposed: Boolean = false,
): SparseMatrix {
    val once = transposeRaw(rows, cols, colPtr, rowIdx, values)
    if (transposed) return once
    return transposeCsc(once)
}
