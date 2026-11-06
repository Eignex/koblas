package com.eignex.koblas.internal.storage

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.sparse.transposeCsc

internal fun transposeMatrixStorage(source: Matrix): Matrix {
    if (source is SparseMatrix) return transposeCsc(source)
    val result = DenseMatrix.zero(source.cols, source.rows)
    if (source.rows == 0 || source.cols == 0) return result
    for (j in 0 until source.cols) {
        for (i in 0 until source.rows) {
            result.values[j + i * source.cols] = if (source is DenseMatrixRegion) {
                source.values[source.offset + i + j * source.leadingDimension]
            } else {
                source[i, j]
            }
        }
    }
    return result
}
