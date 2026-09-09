package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix

/** Adds `alpha · A · x` to [y], mirroring only entries stored in [lower]'s selected triangle. */
internal fun symmetricMultiplyVector(alpha: Double, a: SparseMatrix, x: DoubleArray, y: DoubleArray, lower: Boolean) {
    for (j in 0 until a.cols) {
        a.forEachInColumn(j) { i, value ->
            if (if (lower) i >= j else i <= j) {
                y[i] += alpha * (value * x[j])
                if (i != j) y[j] += alpha * (value * x[i])
            }
        }
    }
}

/** Adds a selected-triangle symmetric sparse product on the requested side of dense [b]. */
internal fun symmetricMultiplyMatrix(
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    c: DenseMatrix,
    lower: Boolean,
    right: Boolean,
) {
    if (right) {
        for (j in 0 until a.cols) {
            a.forEachInColumn(j) { i, value ->
                if (if (lower) i >= j else i <= j) {
                    for (r in 0 until b.rows) {
                        c.data[r + i * c.rows] += alpha * (b.data[r + j * b.rows] * value)
                        if (i != j) c.data[r + j * c.rows] += alpha * (b.data[r + i * b.rows] * value)
                    }
                }
            }
        }
    } else {
        for (j in 0 until a.cols) {
            a.forEachInColumn(j) { i, value ->
                if (if (lower) i >= j else i <= j) {
                    for (column in 0 until b.cols) {
                        c.data[i + column * c.rows] += alpha * (value * b.data[j + column * b.rows])
                        if (i != j) c.data[j + column * c.rows] += alpha * (value * b.data[i + column * b.rows])
                    }
                }
            }
        }
    }
}
