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
 *
 * When [readValues] is false only the pattern is copied and the result's values remain zero, so a zero
 * multiplier can orient an operand without reading its coefficients or allocating a zeroed source copy.
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
