package com.eignex.koblas.internal.storage

import com.eignex.koblas.SparseMatrix

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
