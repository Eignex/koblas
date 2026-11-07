package com.eignex.koblas.internal.storage

import com.eignex.koblas.*

internal fun copyVectorStorage(source: Vector): Vector = when (source) {
    is SparseVector -> SparseVector.wrap(
        source.size,
        source.copyIndices(),
        source.values.copyOfRange(source.offset, source.offset + source.nnz),
    )
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

    is DenseMatrix -> copyDenseMatrix(source)

    else -> {
        val result = DenseMatrix.zero(source.rows, source.cols)
        if (source.rows > 0 && source.cols > 0) {
            for (j in 0 until source.cols) for (i in 0 until source.rows) result[i, j] = source[i, j]
        }
        result
    }
}

internal fun copyDenseMatrix(source: DenseMatrix): DenseMatrix {
    val result = DenseMatrix.zero(source.rows, source.cols)
    copyDenseColumns(source, result)
    return result
}

/** The caller supplies equal shapes and stages destructive aliases before copying column windows. */
internal fun copyDenseColumns(source: DenseMatrix, destination: DenseMatrix) {
    if (source.rows == 0 || source.cols == 0) return
    for (j in 0 until source.cols) {
        val start = source.offset + j * source.leadingDimension
        source.values.copyInto(
            destination.values,
            destination.offset + j * destination.leadingDimension,
            start,
            start + source.rows,
        )
    }
}
