package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.copyMatrixStorage
import com.eignex.koblas.internal.storage.copyVectorStorage

internal fun KoblasEngine.copyIntoVector(source: Vector, destination: Vector) {
    requireSameSize(source.size, destination.size, "copyFrom")
    if (source === destination || source.size == 0) return
    when (destination) {
        is DenseVector -> copyVector(source, destination)

        is SparseVector -> {
            val snapshot = if (source is SparseVector) copyVectorStorage(source) as SparseVector else sparseCopy(source)
            if (snapshot.indices.contentEquals(destination.indices)) {
                snapshot.values.copyInto(destination.values)
            } else {
                destination.replaceStorage(snapshot.indices, snapshot.values)
            }
        }

        else -> {
            val snapshot = copyVectorStorage(source)
            if (snapshot is SparseVector) {
                for (i in 0 until destination.size) destination[i] = 0.0
                snapshot.forEachStored { i, value -> destination[i] = value }
            } else {
                for (i in 0 until destination.size) destination[i] = snapshot[i]
            }
        }
    }
}

internal fun copyIntoMatrix(source: Matrix, destination: Matrix) {
    requireSameShape(source.rows, source.cols, destination, "copyFrom")
    if (source === destination || source.rows == 0 || source.cols == 0) return
    if (destination is SparseMatrix) {
        val snapshot = if (source is SparseMatrix) copyMatrixStorage(source) as SparseMatrix else sparseCopy(source)
        val samePattern = snapshot.colPointers.contentEquals(destination.colPointers) &&
            snapshot.rowIndices.contentEquals(destination.rowIndices)
        if (samePattern) {
            snapshot.values.copyInto(destination.values)
        } else {
            destination.replaceStorage(snapshot.colPointers, snapshot.rowIndices, snapshot.values)
        }
        return
    }
    val snapshot = if (source is DenseMatrix && destination is DenseMatrix &&
        source.values !== destination.values
    ) {
        source
    } else {
        copyMatrixStorage(source)
    }
    if (snapshot is SparseMatrix) {
        for (j in 0 until destination.cols) for (i in 0 until destination.rows) destination[i, j] = 0.0
        for (j in 0 until snapshot.cols) snapshot.forEachInColumn(j) { i, value -> destination[i, j] = value }
    } else if (snapshot is DenseMatrix && destination is DenseMatrix) {
        if (source.rows == 0) return
        for (j in 0 until snapshot.cols) {
            val start = snapshot.offset + j * snapshot.leadingDimension
            snapshot.values.copyInto(
                destination.values,
                destination.offset + j * destination.leadingDimension,
                start,
                start + snapshot.rows,
            )
        }
    } else {
        for (j in 0 until destination.cols) for (i in 0 until destination.rows) destination[i, j] = snapshot[i, j]
    }
}

private fun sparseCopy(source: Vector): SparseVector {
    val indices = mutableListOf<Int>()
    val values = mutableListOf<Double>()
    for (i in 0 until source.size) {
        val value = source[i]
        if (value.toBits() != 0L) {
            indices += i
            values += value
        }
    }
    return SparseVector.wrap(source.size, indices.toIntArray(), values.toDoubleArray())
}

private fun sparseCopy(source: Matrix): SparseMatrix = SparseMatrix.ofColumns(
    source.rows,
    source.cols,
    List(source.cols) { j ->
        buildList {
            for (i in 0 until source.rows) {
                val value = source[i, j]
                if (value.toBits() != 0L) add(i to value)
            }
        }
    },
)
