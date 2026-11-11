package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.vectorProviderFor
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import com.eignex.koblas.internal.storage.*
import com.eignex.koblas.internal.storage.dense.*
import com.eignex.koblas.internal.storage.sparse.*

internal fun KoblasEngine.copyIntoVector(source: Vector, destination: Vector) {
    requireSameSize(source.size, destination.size, "copyFrom")
    if (source === destination || source.size == 0) return
    when (destination) {
        is DenseVector -> copyIntoDenseVector(source, destination)

        is SparseVector -> {
            if (source is SparseVector && source.hasSamePattern(destination)) {
                source.values.copyInto(
                    destination.values,
                    destination.offset,
                    source.offset,
                    source.offset + source.nnz,
                )
            } else {
                val snapshot = if (source is SparseVector) {
                    copyVectorStorage(source) as SparseVector
                } else {
                    sparseCopy(source)
                }
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

private fun SparseVector.hasSamePattern(other: SparseVector): Boolean {
    if (nnz != other.nnz) return false
    val start = offset
    val otherStart = other.offset
    for (k in 0 until nnz) if (indices[start + k] != other.indices[otherStart + k]) return false
    return true
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
            stable.values,
            stable.offset,
            destination.values,
            destination.offset,
            source.size,
            stable.stride,
            destination.stride,
        )
        return
    }
    when (val stable = source.stableFor(destination)) {
        is SparseVector -> {
            applyBeta(this, destination.values, destination.offset, destination.size, 0.0, destination.stride)
            scatterStored(stable, destination)
        }

        is DenseVector -> kernels.copy(
            stable.values,
            stable.offset,
            destination.values,
            destination.offset,
            stable.size,
            stable.stride,
            destination.stride,
        )
    }
}

internal fun copyIntoMatrix(source: Matrix, destination: Matrix) {
    requireSameShape(source.rows, source.cols, destination, "copyFrom")
    if (source === destination || source.rows == 0 || source.cols == 0) return
    if (destination is SparseMatrix) {
        if (source is SparseMatrix && source.colPointers.contentEquals(destination.colPointers) &&
            source.rowIndices.contentEquals(destination.rowIndices)
        ) {
            source.values.copyInto(destination.values)
            return
        }
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
    if (source is DenseMatrix && destination is DenseMatrix && sameRegion(source, destination)) return
    val snapshot = when {
        source is DenseMatrix && destination is DenseMatrix && !regionsOverlap(source, destination) -> source
        source is SparseMatrix && destination is DenseMatrix && source.values !== destination.values -> source
        else -> copyMatrixStorage(source)
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
    val entries = SparseEntries()
    for (i in 0 until source.size) {
        val value = source[i]
        if (value.toBits() != 0L) {
            entries.add(i, value)
        }
    }
    return entries.toVector(source.size)
}

/** Column-major readers already supply unique ascending rows, so construction needs no sorting or merging. */
private fun sparseCopy(source: Matrix): SparseMatrix {
    val pointers = IntArray(pointerLength(source.cols, "copyFrom"))
    val entries = SparseEntries()
    for (j in 0 until source.cols) {
        for (i in 0 until source.rows) {
            val value = source[i, j]
            if (value.toBits() != 0L) {
                entries.add(i, value)
            }
        }
        pointers[j + 1] = entries.size
    }
    return entries.toMatrix(source.rows, source.cols, pointers)
}
