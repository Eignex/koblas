package com.eignex.koblas.sparse.internal

import com.eignex.koblas.SparseMatrix

/** Merges CSC columns for `alpha · A + B`, retaining the complete structural union. */
internal fun addScaledCsc(alpha: Double, a: SparseMatrix, b: SparseMatrix): SparseMatrix {
    val pointers = IntArray(a.cols + 1)
    var rows = IntArray(a.nnz + b.nnz)
    var values = DoubleArray(rows.size)
    var count = 0
    for (j in 0 until a.cols) {
        count += SparseAccumulationKernels.mergeScaledColumns(
            alpha,
            a.rowIdx,
            a.values,
            a.colPtr[j],
            a.colPtr[j + 1],
            b.rowIdx,
            b.values,
            b.colPtr[j],
            b.colPtr[j + 1],
            rows,
            values,
            count,
        )
        pointers[j + 1] = count
    }
    return SparseMatrix.wrapTrusted(a.rows, a.cols, pointers, rows.copyOf(count), values.copyOf(count))
}
