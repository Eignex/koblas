package com.eignex.koblas.sparse.internal

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi

/**
 * The CSC transpose, which is also the CSC-to-CSR conversion. Explicitly stored zeros survive, since the
 * transpose is structural.
 *
 * The result holds the CSC invariant by construction rather than by checking: the walk visits source columns
 * in order, so each output column collects its entries by ascending source column, and an input with no
 * repeated coordinate yields no repeated row.
 */
@OptIn(UnsafeKoblasApi::class)
internal fun transposeCsc(a: SparseMatrix): SparseMatrix {
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
    val next = outPointers.copyOf()
    for (j in 0 until cols) {
        for (k in colPointers[j] until colPointers[j + 1]) {
            val slot = next[rowIndices[k]]++
            outIndices[slot] = j
            outValues[slot] = values[k]
        }
    }
    return SparseMatrix.wrapTrusted(cols, rows, outPointers, outIndices, outValues)
}
