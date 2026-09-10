package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.absoluteSum
import kotlin.math.abs

/** Materializes a column-major matrix as independent logical rows. */
internal fun denseRows(data: DoubleArray, rows: Int, columns: Int): Array<DoubleArray> =
    Array(rows) { row -> DoubleArray(columns) { column -> data[row + column * rows] } }

/** Copies logical rows into a column-major destination. */
internal fun flattenDenseRows(rows: Array<DoubleArray>, columns: Int, destination: DoubleArray) {
    for (row in rows.indices) {
        for (column in 0 until columns) destination[row + column * rows.size] = rows[row][column]
    }
}

/** Copies logical columns into a column-major destination. */
internal fun flattenDenseColumns(columns: Array<DoubleArray>, rows: Int, destination: DoubleArray) {
    for (column in columns.indices) columns[column].copyInto(destination, column * rows)
}

/** Writes one constant to the diagonal of a zeroed square matrix. */
internal fun fillDenseDiagonal(data: DoubleArray, order: Int, value: Double) {
    for (index in 0 until order) data[index + index * order] = value
}

/** Copies diagonal entries into a zeroed square matrix. */
internal fun fillDenseDiagonal(data: DoubleArray, values: DoubleArray) {
    val order = values.size
    for (index in values.indices) data[index + index * order] = values[index]
}

/** Copies one logical row from a column-major matrix into a contiguous destination. */
internal fun gatherDenseRow(data: DoubleArray, rows: Int, columns: Int, row: Int, destination: DoubleArray) {
    for (column in 0 until columns) destination[column] = data[row + column * rows]
}

/** Scales every column-major entry by the factor for its logical row. */
internal fun scaleDenseRows(data: DoubleArray, rows: Int, columns: Int, factors: DoubleArray) {
    for (column in 0 until columns) {
        val offset = column * rows
        for (row in 0 until rows) data[offset + row] *= factors[row]
    }
}

/** Clears the strict upper triangle while retaining positive-zero writes. */
internal fun clearDenseStrictUpper(data: DoubleArray, rows: Int, columns: Int) {
    for (column in 1 until columns) {
        val start = column * rows
        data.fill(0.0, start, start + minOf(column, rows))
    }
}

/** Maximum absolute column sum, carrying any NaN entry through the reduction. */
internal fun denseMatrixNorm1(data: DoubleArray, rows: Int, columns: Int): Double {
    var maximum = 0.0
    for (column in 0 until columns) {
        maximum = carryingMaximum(maximum, absoluteSum(data, column * rows, rows))
    }
    return maximum
}

/** Maximum absolute row sum using caller-owned zeroed scratch, carrying any NaN through. */
internal fun denseMatrixNormInf(data: DoubleArray, rows: Int, columns: Int, sums: DoubleArray): Double {
    for (column in 0 until columns) {
        val offset = column * rows
        for (row in 0 until rows) sums[row] += abs(data[offset + row])
    }
    return maximumCarryingNaN(sums, rows)
}

/** Maximum over a contiguous prefix, with NaN winning over every prior value. */
internal fun maximumCarryingNaN(values: DoubleArray, size: Int): Double {
    var maximum = 0.0
    for (index in 0 until size) maximum = carryingMaximum(maximum, values[index])
    return maximum
}

private fun carryingMaximum(current: Double, candidate: Double): Double =
    if (candidate > current || candidate.isNaN()) candidate else current
