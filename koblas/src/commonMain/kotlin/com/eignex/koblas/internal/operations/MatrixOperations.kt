package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.scalar.scalarMatrixNorm
import com.eignex.koblas.internal.kernels.scalar.scalarMatrixSum
import com.eignex.koblas.internal.kernels.scalar.scalarMatrixSums
import com.eignex.koblas.internal.kernels.scalar.sparse.sparseMatrixInfinityNorm
import kotlin.math.hypot

/** Storage adaptation for reductions; dense windows and packed CSC values reach the selected sum kernel. */
internal fun KoblasEngine.sumMatrix(a: Matrix): Double = when (a) {
    is DenseMatrix -> {
        when {
            a.rows == 0 || a.cols == 0 -> 0.0

            a.leadingDimension == a.rows -> sum(DenseVector.wrap(a.values, a.offset, a.rows * a.cols))

            else -> {
                var total = 0.0
                for (j in 0 until a.cols) {
                    total += sum(a.columnView(j))
                }
                total
            }
        }
    }

    is SparseMatrix -> sum(DenseVector.wrap(a.values, 0, a.values.size))

    else -> scalarMatrixSum(a)
}

internal fun KoblasEngine.sumMatrix(a: Matrix, axis: Int): DoubleArray {
    require(axis == 0 || axis == 1) { "sum axis must be zero or one, got $axis" }
    if (a.rows == 0 || a.cols == 0) return DoubleArray(if (axis == 0) a.cols else a.rows)
    return when {
        a is DenseMatrix -> if (axis == 0) {
            DoubleArray(a.cols) { j -> sum(DenseVector.wrap(a.values, a.offset + j * a.leadingDimension, a.rows)) }
        } else {
            DoubleArray(a.rows) { i -> sum(DenseVector.wrap(a.values, a.offset + i, a.cols, a.leadingDimension)) }
        }

        a is SparseMatrix && axis == 0 -> DoubleArray(a.cols) { j ->
            val start = a.colPointers[j]
            sum(DenseVector.wrap(a.values, start, a.colPointers[j + 1] - start))
        }

        else -> scalarMatrixSums(a, axis)
    }
}

internal fun KoblasEngine.normMatrix(a: Matrix, type: Norm = Norm.Frobenius): Double {
    require(type != Norm.Two) { "Spectral matrix norm is not supported" }
    if (a.rows == 0 || a.cols == 0) return 0.0
    return when (a) {
        is DenseMatrix -> denseMatrixNorm(a, type)
        is SparseMatrix -> sparseMatrixNorm(a, type)
        else -> scalarMatrixNorm(a, type)
    }
}

private fun KoblasEngine.denseMatrixNorm(a: DenseMatrix, type: Norm): Double {
    if (type == Norm.Frobenius && a.leadingDimension == a.rows) {
        return nrm2(DenseVector.wrap(a.values, a.offset, a.rows * a.cols))
    }
    val rows = type == Norm.Infinity
    val count = if (rows) a.rows else a.cols
    val length = if (rows) a.cols else a.rows
    val stride = if (rows) a.leadingDimension else 1
    var result = 0.0
    for (i in 0 until count) {
        val offset = a.offset + if (rows) i else i * a.leadingDimension
        result = when (type) {
            Norm.Frobenius -> hypot(result, nrm2(DenseVector.wrap(a.values, offset, length, stride)))
            Norm.MaxAbs -> maxOf(result, maximumMagnitude(a.values, offset, length, stride))
            Norm.One, Norm.Infinity -> maxOf(result, asum(DenseVector.wrap(a.values, offset, length, stride)))
            Norm.Two -> error("Spectral matrix norm is not supported")
        }
    }
    return result
}

private fun KoblasEngine.sparseMatrixNorm(a: SparseMatrix, type: Norm): Double = when (type) {
    Norm.Frobenius -> nrm2(DenseVector.wrap(a.values, 0, a.values.size))

    Norm.MaxAbs -> maximumMagnitude(a.values, 0, a.values.size, 1)

    Norm.Infinity -> sparseMatrixInfinityNorm(a)

    Norm.One -> {
        var result = 0.0
        for (j in 0 until a.cols) {
            val start = a.colPointers[j]
            result = maxOf(result, asum(DenseVector.wrap(a.values, start, a.colPointers[j + 1] - start)))
        }
        result
    }

    Norm.Two -> error("Spectral matrix norm is not supported")
}
