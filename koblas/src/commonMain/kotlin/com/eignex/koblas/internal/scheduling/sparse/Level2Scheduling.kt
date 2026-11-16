@file:Suppress("LongParameterList") // Validated sparse matrix-vector windows.

package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.sparse.sparseSymmetricVectorColumn
import com.eignex.koblas.internal.operations.applyBeta

@Suppress("LongParameterList") // the BLAS dgemv signature, over the snapshots taken for it
internal fun KernelProvider.scheduleSparseGemv(
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aColumns: Int,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean,
) {
    applyBeta(this, y, 0, y.size, beta)
    if (transpose) {
        for (j in 0 until aColumns) {
            val start = aPointers[j]
            val end = aPointers[j + 1]
            if (start == end) continue
            val sum = dot(
                aIndices,
                start,
                aValues,
                start,
                end - start,
                x,
            )
            y[j] += alpha * sum
        }
    } else {
        for (j in 0 until aColumns) {
            val xj = alpha * x[j]
            val start = aPointers[j]
            this.axpy(
                aIndices,
                start,
                aValues,
                start,
                aPointers[j + 1] - start,
                xj,
                y,
            )
        }
    }
}

internal fun KernelProvider.scheduleSparseSymv(
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aColumns: Int,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean,
) {
    applyBeta(this, y, 0, y.size, beta)
    for (column in 0 until aColumns) {
        sparseSymmetricVectorColumn(
            alpha,
            column,
            aIndices,
            aValues,
            aPointers[column],
            aPointers[column + 1],
            x,
            y,
            lower,
        )
    }
}
