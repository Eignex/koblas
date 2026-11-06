package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.sparse.indexedScatter
import com.eignex.koblas.internal.storage.isWholeArray

/** Sparse schedules consume contiguous vectors; view packing preserves their no-read and alias rules. */
@Suppress("LongParameterList") // BLAS operands, flags and scratch
internal fun KoblasEngine.sparseMatrixVector(
    alpha: Double,
    a: SparseMatrix,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    flag: Boolean,
    scratch: Scratch?,
    symmetric: Boolean,
) {
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
        applyBeta(this, y.values, y.offset, y.size, beta, y.stride)
        return
    }
    val input = if (x.isWholeArray) x.values else x.toDoubleArray()
    val output = when {
        y.isWholeArray -> y.values
        beta == 0.0 -> DoubleArray(y.size)
        else -> y.toDoubleArray()
    }
    if (symmetric) {
        kernels.symv(alpha, a, input, beta, output, flag, scratch)
    } else {
        kernels.gemv(alpha, a, input, beta, output, flag, scratch)
    }
    if (!y.isWholeArray) for (i in 0 until y.size) y[i] = output[i]
}

@Suppress("LongParameterList") // triangle flags and scratch
internal fun KoblasEngine.sparseTriangularVector(
    a: SparseMatrix,
    x: DenseVector,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    scratch: Scratch?,
    solve: Boolean,
) {
    val values = if (x.isWholeArray) x.values else x.toDoubleArray()
    if (solve) {
        kernels.trsv(a, values, lower, transpose, unitDiag, scratch)
    } else {
        kernels.trmv(a, values, lower, transpose, unitDiag, scratch)
    }
    if (!x.isWholeArray) for (i in 0 until x.size) x[i] = values[i]
}

internal fun KoblasEngine.scatterVector(source: SparseVector, destination: DenseVector) {
    requireSameSize(source.size, destination.size, "scatter")
    val aliased = source.values === destination.values
    val values = if (aliased) source.values.copyOfRange(source.offset, source.offset + source.nnz) else source.values
    val valueOffset = if (aliased) 0 else source.offset
    if (destination.isWholeArray) {
        kernels.scatter(source.indices, source.offset, values, valueOffset, source.nnz, destination.values)
    } else {
        indexedScatter(
            source.indices,
            source.offset,
            values,
            valueOffset,
            source.nnz,
            destination.values,
            destination.offset,
            destination.stride,
        )
    }
}

internal fun KoblasEngine.gatherVector(source: DenseVector, destination: SparseVector, zero: Boolean) {
    requireSameSize(source.size, destination.size, "gather")
    require(source.values !== destination.values) { "gather requires distinct coefficient buffers" }
    if (source.isWholeArray) {
        if (zero) {
            kernels.gatherZero(
                destination.indices, destination.offset, destination.values, destination.offset, destination.nnz, source.values,
            )
        } else {
            kernels.gather(
                destination.indices, destination.offset, destination.values, destination.offset, destination.nnz, source.values,
            )
        }
    } else {
        for (k in destination.offset until destination.offset + destination.nnz) {
            val i = destination.indices[k]
            destination.values[k] = source[i]
            if (zero) source[i] = 0.0
        }
    }
}
