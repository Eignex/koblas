@file:Suppress("LongParameterList") // Validated sparse matrix-vector windows.

package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.portable.dense.applyBeta

@Suppress("LongParameterList") // the BLAS dgemv signature, over the snapshots taken for it
internal fun KoblasEngine.portableSparseGemv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean,
) {
    applyBeta(this, y, 0, y.size, beta)
    if (transpose) {
        for (j in 0 until a.cols) {
            val start = a.colPointers[j]
            val end = a.colPointers[j + 1]
            if (start == end) continue
            val sum = this.dot(
                a.rowIndices,
                start,
                a.values,
                start,
                end - start,
                x,
            )
            y[j] += alpha * sum
        }
    } else {
        for (j in 0 until a.cols) {
            val xj = alpha * x[j]
            val start = a.colPointers[j]
            this.axpy(
                a.rowIndices,
                start,
                a.values,
                start,
                a.colPointers[j + 1] - start,
                xj,
                y,
            )
        }
    }
}

internal fun KoblasEngine.portableSparseSymv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean,
) {
    applyBeta(this, y, 0, y.size, beta)
    for (column in 0 until a.cols) {
        sparseSymmetricVectorColumn(
            alpha,
            column,
            a.rowIndices,
            a.values,
            a.colPointers[column],
            a.colPointers[column + 1],
            x,
            y,
            lower,
        )
    }
}
