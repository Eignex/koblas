package com.eignex.koblas.internal.storage

import com.eignex.koblas.SparseMatrix

/** Replacing a column preserves every other column and adjusts the following CSC boundaries. */
internal fun SparseMatrix.replaceColumn(column: Int, indices: IntArray, coefficients: DoubleArray) {
    val start = colPointers[column]
    val end = colPointers[column + 1]
    val count = nnz.toLong() - (end - start) + indices.size
    requireShape(count <= Int.MAX_VALUE) { "stored entry count exceeds Int capacity" }
    val pointers = colPointers.copyOf()
    val rows = IntArray(count.toInt())
    val values = DoubleArray(count.toInt())
    rowIndices.copyInto(rows, 0, 0, start)
    this.values.copyInto(values, 0, 0, start)
    indices.copyInto(rows, start)
    coefficients.copyInto(values, start)
    rowIndices.copyInto(rows, start + indices.size, end)
    this.values.copyInto(values, start + indices.size, end)
    val delta = indices.size - (end - start)
    for (j in column + 1..cols) pointers[j] += delta
    replaceStorage(pointers, rows, values)
}
