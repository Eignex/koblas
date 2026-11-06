package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.scalarMatrixSum
import com.eignex.koblas.internal.kernels.scalar.scalarMatrixSums

/** Storage adaptation for reductions; dense windows and packed CSC values reach the selected sum kernel. */
internal fun KoblasEngine.sumMatrix(a: Matrix): Double = when (a) {
    is DenseMatrixRegion -> {
        when {
            a.rows == 0 || a.cols == 0 -> 0.0

            a.leadingDimension == a.rows -> sum(a.values, a.offset, a.rows * a.cols)

            else -> {
                var total = 0.0
                for (j in 0 until a.cols) total += sum(a.values, a.offset + j * a.leadingDimension, a.rows)
                total
            }
        }
    }

    is SparseMatrix -> sum(a.values, 0, a.values.size)

    else -> scalarMatrixSum(a)
}

internal fun KoblasEngine.sumMatrix(a: Matrix, axis: Int): DoubleArray {
    require(axis == 0 || axis == 1) { "sum axis must be zero or one, got $axis" }
    if (a.rows == 0 || a.cols == 0) return DoubleArray(if (axis == 0) a.cols else a.rows)
    return when {
        a is DenseMatrixRegion -> if (axis == 0) {
            DoubleArray(a.cols) { j -> sum(a.values, a.offset + j * a.leadingDimension, a.rows) }
        } else {
            DoubleArray(a.rows) { i -> sum(a.values, a.offset + i, a.cols, a.leadingDimension) }
        }

        a is SparseMatrix && axis == 0 -> DoubleArray(a.cols) { j ->
            val start = a.colPointers[j]
            sum(a.values, start, a.colPointers[j + 1] - start)
        }

        else -> scalarMatrixSums(a, axis)
    }
}
