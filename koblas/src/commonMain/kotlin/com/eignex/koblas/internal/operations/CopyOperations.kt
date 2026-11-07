package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.vectorProviderFor
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import com.eignex.koblas.internal.kernels.scalar.zeroVector
import com.eignex.koblas.internal.kernels.scalar.sparse.indexedScatter
import com.eignex.koblas.internal.storage.*
import com.eignex.koblas.internal.storage.sparse.pointerLength

internal fun KoblasEngine.copyIntoVector(source: Vector, destination: Vector) {
    requireSameSize(source.size, destination.size, "copyFrom")
    if (source === destination || source.size == 0) return
    when (destination) {
        is DenseVector -> copyIntoDenseVector(source, destination)

        is SparseVector -> {
            val snapshot = if (source is SparseVector) copyVectorStorage(source) as SparseVector else sparseCopy(source)
            destination.replaceStorage(snapshot.indices, snapshot.values)
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

/** The caller validates size; aliases must be staged before clearing a dense destination. */
private fun KoblasEngine.copyIntoDenseVector(source: Vector, destination: DenseVector) {
    if (source is DenseVector) {
        if (sameVectorMapping(source, destination)) return
        val contiguous = source.stride == 1 && destination.stride == 1
        val provider = vectorProviderFor(EngineImplementation.Copy, source.size, contiguous)
        // Portable contiguous copy is a memmove, so it needs no alias snapshot.
        val stable = if (provider === ScalarEngine && contiguous) source else source.stableFor(destination)
        provider.kernels.copy(
            stable.values, stable.offset, destination.values, destination.offset,
            source.size, stable.stride, destination.stride,
        )
        return
    }
    when (val stable = source.stableFor(destination)) {
        is SparseVector -> {
            zeroVector(destination.values, destination.offset, destination.size, destination.stride)
            if (destination.isWholeArray) {
                kernels.scatter(
                    stable.indices, stable.offset, stable.values, stable.offset, stable.nnz, destination.values,
                )
            } else {
                indexedScatter(
                    stable.indices, stable.offset, stable.values, stable.offset, stable.nnz,
                    destination.values, destination.offset, destination.stride,
                )
            }
        }

        is DenseVector -> kernels.copy(
            stable.values, stable.offset, destination.values, destination.offset,
            stable.size, stable.stride, destination.stride,
        )
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
        copyDenseColumns(snapshot, destination)
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

/** Column-major readers already supply unique ascending rows, so construction needs no sorting or merging. */
private fun sparseCopy(source: Matrix): SparseMatrix {
    val pointers = IntArray(pointerLength(source.cols, "copyFrom"))
    val indices = mutableListOf<Int>()
    val values = mutableListOf<Double>()
    for (j in 0 until source.cols) {
        for (i in 0 until source.rows) {
            val value = source[i, j]
            if (value.toBits() != 0L) {
                indices += i
                values += value
            }
        }
        pointers[j + 1] = values.size
    }
    return SparseMatrix.wrapTrusted(source.rows, source.cols, pointers, indices.toIntArray(), values.toDoubleArray())
}
