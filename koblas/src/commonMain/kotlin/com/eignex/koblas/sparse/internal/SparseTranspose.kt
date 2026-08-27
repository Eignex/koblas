package com.eignex.koblas.sparse.internal

import com.eignex.koblas.core.F64SparseMatrix

/**
 * The CSC transpose, which is also the CSC-to-CSR conversion. Explicitly stored zeros survive, since the
 * transpose is structural.
 *
 * Its own function rather than a method on the portable backend because the portable factorizations
 * transpose too, and reaching the seam for it would route the definition of a routine through whichever
 * backend happens to be registered.
 */
internal fun transposeCsc(a: F64SparseMatrix): F64SparseMatrix =
    transposeRaw(a.rows, a.cols, a.colPtr, a.rowIdx, a.values)

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
): F64SparseMatrix {
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
    return F64SparseMatrix.wrap(cols, rows, outPtr, outIdx, outVal)
}
