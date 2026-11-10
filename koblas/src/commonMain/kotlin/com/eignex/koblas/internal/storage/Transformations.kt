package com.eignex.koblas.internal.storage

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.dense.*
import com.eignex.koblas.internal.storage.sparse.transposeCsc

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

internal fun transposeMatrixStorage(source: Matrix): Matrix {
    if (source is SparseMatrix) return transposeCsc(source)
    val result = DenseMatrix.zero(source.cols, source.rows)
    if (source.rows == 0 || source.cols == 0) return result
    for (j in 0 until source.cols) {
        for (i in 0 until source.rows) {
            result.values[j + i * source.cols] = if (source is DenseMatrix) {
                source.values[source.offset + i + j * source.leadingDimension]
            } else {
                source[i, j]
            }
        }
    }
    return result
}

/** Opaque matrix storage exposes its column through the same indexed mutable contract. */
internal class MatrixColumnVector(private val matrix: Matrix, private val column: Int) : Vector {
    override val size: Int get() = matrix.rows

    override fun get(i: Int): Double = matrix[i, column]

    override fun set(i: Int, v: Double) {
        matrix[i, column] = v
    }
}
