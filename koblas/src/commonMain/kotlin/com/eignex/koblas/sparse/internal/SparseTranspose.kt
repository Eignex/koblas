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
internal fun transposeCsc(a: F64SparseMatrix): F64SparseMatrix {
    val outPtr = IntArray(a.rows + 1)
    // Counts of row i land in outPtr(i + 1), then a prefix sum turns them into offsets.
    for (k in a.rowIdx.indices) outPtr[a.rowIdx[k] + 1]++
    for (i in 0 until a.rows) outPtr[i + 1] += outPtr[i]
    val outIdx = IntArray(a.values.size)
    val outVal = DoubleArray(a.values.size)
    // Walking source columns in order leaves each destination column's row indices ascending.
    val next = outPtr.copyOf()
    for (j in 0 until a.cols) {
        for (k in a.colPtr[j] until a.colPtr[j + 1]) {
            val slot = next[a.rowIdx[k]]++
            outIdx[slot] = j
            outVal[slot] = a.values[k]
        }
    }
    return F64SparseMatrix.wrap(a.cols, a.rows, outPtr, outIdx, outVal)
}
