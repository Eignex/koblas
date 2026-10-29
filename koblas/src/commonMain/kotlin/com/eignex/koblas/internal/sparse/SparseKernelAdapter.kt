package com.eignex.koblas.internal.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.VectorRoute
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.diagnostics.vectorRoute
import com.eignex.koblas.requireIndex
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.*

internal class SparseKernelAdapter(
    override val name: String,
    private val denseVectorKernels: DenseVectorKernels,
    private val indexedSparseKernels: IndexedSparseKernels,
) : SparseKernels {
    override fun routeOf(operation: SparseOperation, count: Int): VectorRoute<SparseOperation> {
        // Whole-support reductions use contiguous dense kernels; the other operations use indexed kernels.
        if (operation == SparseOperation.Nrm2 || operation == SparseOperation.Asum) {
            val dense = if (operation == SparseOperation.Nrm2) DenseOperation.Nrm2 else DenseOperation.Asum
            return vectorRoute(
                operation,
                operation.entryPoint,
                count,
                denseVectorKernels.name,
                denseVectorKernels.implementationFor(dense, count),
                "a contiguous run of stored values",
            )
        }
        return vectorRoute(
            operation,
            operation.entryPoint,
            count,
            indexedSparseKernels.name,
            indexedSparseKernels.implementationFor(operation, count),
        )
    }

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return indexedSparseKernels.dotDense(x.indices, x.values, y)
    }

    override fun dotColumn(matrix: SparseMatrix, column: Int, dense: DoubleArray): Double {
        validateColumnDot(matrix, column, dense)
        val start = matrix.colPointers[column]
        val count = matrix.colPointers[column + 1] - start
        return indexedSparseKernels.dotDense(matrix.rowIndices, start, matrix.values, start, count, dense)
    }

    override fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double {
        requireSparseWindow(indices.size, indexOffset, count, "indices")
        requireSparseWindow(values.size, valueOffset, count, "values")
        validateSparseIndices(indices, indexOffset, count, dense.size, strictlyIncreasing = false)
        return indexedSparseKernels.dotDense(indices, indexOffset, values, valueOffset, count, dense)
    }

    override fun dot(x: SparseVector, y: SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return indexedSparseKernels.dotSparse(x.indices, x.values, y.indices, y.values)
    }

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
        if (alpha == 0.0) return
        indexedSparseKernels.axpy(x.indices, x.values, alpha, y)
    }

    override fun axpy(
        destination: DoubleArray,
        alpha: Double,
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
    ) {
        validateMutation(indices, indexOffset, values, valueOffset, count, destination)
        if (alpha == 0.0) return
        indexedSparseKernels.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
    }

    override fun scatter(x: SparseVector, out: DoubleArray) {
        requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
        indexedSparseKernels.scatter(x.indices, x.values, out)
    }

    override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        validateMutation(indices, indexOffset, values, valueOffset, count, destination)
        indexedSparseKernels.scatter(indices, indexOffset, values, valueOffset, count, destination)
    }

    override fun gather(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gather: sizes differ, ${x.size} vs ${from.size}" }
        require(x.values !== from) { "values and source must use distinct buffers" }
        indexedSparseKernels.gather(x.indices, x.values, from)
    }

    override fun gatherZero(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gatherZero: sizes differ, ${x.size} vs ${from.size}" }
        require(x.values !== from) { "values and source must use distinct buffers" }
        indexedSparseKernels.gatherZero(x.indices, x.values, from)
    }

    override fun nrm2(x: SparseVector): Double = denseVectorKernels.nrm2(x.values, 0, x.values.size)

    override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
        requireSparseWindow(indices.size, indexOffset, count, "indices")
        validateSparseIndices(indices, indexOffset, count, values.size, strictlyIncreasing = false)
        return indexedSparseKernels.nrm2(indices, indexOffset, count, values)
    }

    override fun asum(x: SparseVector): Double = denseVectorKernels.asum(x.values, 0, x.values.size)

    private fun validateMutation(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        requireSparseWindow(indices.size, indexOffset, count, "indices")
        requireSparseWindow(values.size, valueOffset, count, "values")
        require(values !== destination) { "values and destination must use distinct buffers" }
        validateSparseIndices(indices, indexOffset, count, destination.size, strictlyIncreasing = true)
    }
}

internal fun validateColumnDot(matrix: SparseMatrix, column: Int, dense: DoubleArray) {
    requireIndex(column in 0 until matrix.cols) { "column $column is outside [0, ${matrix.cols})" }
    requireShape(matrix.rows == dense.size) { "dotColumn: sizes differ, ${matrix.rows} vs ${dense.size}" }
}

private fun requireSparseWindow(length: Int, offset: Int, count: Int, name: String) {
    require(offset >= 0 && count >= 0 && offset.toLong() + count <= length) {
        "$name window [$offset, ${offset.toLong() + count}) exceeds length $length"
    }
}

private fun validateSparseIndices(
    indices: IntArray,
    offset: Int,
    count: Int,
    dimension: Int,
    strictlyIncreasing: Boolean,
) {
    var previous = -1
    for (k in 0 until count) {
        val index = indices[offset + k]
        requireIndex(index in 0 until dimension) { "index $index is outside [0, $dimension)" }
        if (strictlyIncreasing) {
            require(index > previous) { "indices must be strictly increasing" }
            previous = index
        }
    }
}
