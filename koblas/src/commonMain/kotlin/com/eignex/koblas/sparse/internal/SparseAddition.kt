package com.eignex.koblas.sparse.internal

import com.eignex.koblas.SparseMatrix

/** Merges CSC columns for `alpha · A + B`, retaining the complete structural union. */
internal fun addScaledCsc(alpha: Double, a: SparseMatrix, b: SparseMatrix): SparseMatrix {
    val pointers = IntArray(a.cols + 1)
    var rows = IntArray(a.nnz + b.nnz)
    var values = DoubleArray(rows.size)
    var count = 0
    for (j in 0 until a.cols) {
        var ap = a.colPtr[j]
        val ae = a.colPtr[j + 1]
        var bp = b.colPtr[j]
        val be = b.colPtr[j + 1]
        while (ap < ae || bp < be) {
            val ar = if (ap < ae) a.rowIdx[ap] else Int.MAX_VALUE
            val br = if (bp < be) b.rowIdx[bp] else Int.MAX_VALUE
            when {
                ar < br -> {
                    rows[count] = ar
                    values[count] = if (alpha == 0.0) alpha else alpha * a.values[ap]
                    ap++
                }

                br < ar -> {
                    rows[count] = br
                    values[count] = b.values[bp]
                    bp++
                }

                else -> {
                    rows[count] = ar
                    values[count] = if (alpha == 0.0) b.values[bp] else alpha * a.values[ap] + b.values[bp]
                    ap++
                    bp++
                }
            }
            count++
        }
        pointers[j + 1] = count
    }
    return SparseMatrix.wrapTrusted(a.rows, a.cols, pointers, rows.copyOf(count), values.copyOf(count))
}
