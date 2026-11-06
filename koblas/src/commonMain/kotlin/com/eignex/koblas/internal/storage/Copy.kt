package com.eignex.koblas.internal.storage

import com.eignex.koblas.*

internal fun copyVectorStorage(source: Vector): Vector = when (source) {
    is SparseVector -> SparseVector.wrap(source.size, source.indices.copyOf(), source.values.copyOf())
    else -> DenseVector.wrap(source.toDoubleArray())
}

internal fun copyMatrixStorage(source: Matrix): Matrix = when (source) {
    is SparseMatrix -> SparseMatrix.wrapTrusted(
        source.rows,
        source.cols,
        source.colPointers.copyOf(),
        source.rowIndices.copyOf(),
        source.values.copyOf(),
    )

    else -> {
        val result = DenseMatrix.zero(source.rows, source.cols)
        if (source.rows > 0 && source.cols > 0) {
            if (source is DenseMatrix) {
                for (j in 0 until source.cols) {
                    val start = source.offset + j * source.leadingDimension
                    source.values.copyInto(result.values, j * source.rows, start, start + source.rows)
                }
            } else {
                for (j in 0 until source.cols) for (i in 0 until source.rows) result[i, j] = source[i, j]
            }
        }
        result
    }
}
