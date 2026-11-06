package com.eignex.koblas.internal.kernels.scalar

import com.eignex.koblas.Matrix
import com.eignex.koblas.Norm
import com.eignex.koblas.SparseMatrix
import kotlin.math.abs
import kotlin.math.hypot

internal fun scalarMatrixNorm(a: Matrix, type: Norm): Double {
    var result = 0.0
    when (type) {
        Norm.One, Norm.Infinity -> {
            val rows = type == Norm.Infinity
            val count = if (rows) a.rows else a.cols
            val length = if (rows) a.cols else a.rows
            for (i in 0 until count) {
                var sum = 0.0
                for (k in 0 until length) sum += abs(if (rows) a[i, k] else a[k, i])
                result = maxOf(result, sum)
            }
        }

        Norm.Frobenius -> for (j in 0 until a.cols) for (i in 0 until a.rows) result = hypot(result, a[i, j])

        Norm.MaxAbs -> for (j in 0 until a.cols) for (i in 0 until a.rows) result = maxOf(result, abs(a[i, j]))

        Norm.Two -> error("Spectral matrix norm is not supported")
    }
    return result
}

internal fun scalarMatrixSum(a: Matrix): Double {
    var total = 0.0
    for (j in 0 until a.cols) for (i in 0 until a.rows) total += a[i, j]
    return total
}

/** Sparse row accumulation walks CSC support; opaque matrices supply entry readers. */
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
