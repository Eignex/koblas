package com.eignex.koblas.sparse.internal

import com.eignex.koblas.dense.maximumCarryingNaN
import com.eignex.koblas.internal.numeric.absoluteSum
import kotlin.math.abs

/** Materializes CSC storage as independent logical rows. */
internal fun sparseRows(
    rows: Int,
    columns: Int,
    columnPointers: IntArray,
    rowIndices: IntArray,
    values: DoubleArray,
): Array<DoubleArray> = Array(rows) { DoubleArray(columns) }.also { destination ->
    for (column in 0 until columns) {
        for (position in columnPointers[column] until columnPointers[column + 1]) {
            destination[rowIndices[position]][column] = values[position]
        }
    }
}

/** Number of stored entries in one logical CSC row. */
internal fun sparseRowSize(row: Int, columns: Int, columnPointers: IntArray, rowIndices: IntArray): Int {
    var count = 0
    for (column in 0 until columns) {
        for (position in columnPointers[column] until columnPointers[column + 1]) {
            if (rowIndices[position] == row) count++
        }
    }
    return count
}

/** Gathers a logical CSC row into ascending column indices and matching values. */
internal fun gatherSparseRow(
    row: Int,
    columns: Int,
    columnPointers: IntArray,
    rowIndices: IntArray,
    values: DoubleArray,
    destinationIndices: IntArray,
    destinationValues: DoubleArray,
) {
    var destination = 0
    for (column in 0 until columns) {
        for (position in columnPointers[column] until columnPointers[column + 1]) {
            if (rowIndices[position] == row) {
                destinationIndices[destination] = column
                destinationValues[destination] = values[position]
                destination++
            }
        }
    }
}

/** Scales each CSC column without changing its support. */
internal fun scaleSparseColumns(columnPointers: IntArray, values: DoubleArray, columns: Int, factors: DoubleArray) {
    for (column in 0 until columns) {
        val factor = factors[column]
        if (factor == 1.0) continue
        for (position in columnPointers[column] until columnPointers[column + 1]) values[position] *= factor
    }
}

/** Scales each stored CSC entry by the factor for its row. */
internal fun scaleSparseRows(rowIndices: IntArray, values: DoubleArray, factors: DoubleArray) {
    for (position in values.indices) values[position] *= factors[rowIndices[position]]
}

/** Maximum absolute CSC column sum, carrying any stored NaN through the reduction. */
internal fun sparseMatrixNorm1(values: DoubleArray, columnPointers: IntArray, columns: Int): Double {
    var maximum = 0.0
    for (column in 0 until columns) {
        val start = columnPointers[column]
        val candidate = absoluteSum(values, start, columnPointers[column + 1] - start)
        if (candidate > maximum || candidate.isNaN()) maximum = candidate
    }
    return maximum
}

/** Maximum absolute CSC row sum using caller-owned zeroed scratch. */
internal fun sparseMatrixNormInf(rowIndices: IntArray, values: DoubleArray, rows: Int, sums: DoubleArray): Double {
    for (position in values.indices) sums[rowIndices[position]] += abs(values[position])
    return maximumCarryingNaN(sums, rows)
}
