package com.eignex.koblas.internal.kernels.scalar

import com.eignex.koblas.Matrix
import com.eignex.koblas.SparseMatrix

internal fun scalarMatrixSum(a: Matrix): Double {
    var total = 0.0
    for (j in 0 until a.cols) for (i in 0 until a.rows) total += a[i, j]
    return total
}

/** Sparse row accumulation walks CSC support; opaque matrices need only their read-only entry contract. */
internal fun scalarMatrixSums(a: Matrix, axis: Int): DoubleArray {
    val result = DoubleArray(if (axis == 0) a.cols else a.rows)
    for (j in 0 until a.cols) {
        if (a is SparseMatrix) {
            a.forEachInColumn(j) { i, value -> result[if (axis == 0) j else i] += value }
        } else {
            for (i in 0 until a.rows) result[if (axis == 0) j else i] += a[i, j]
        }
    }
    return result
}
