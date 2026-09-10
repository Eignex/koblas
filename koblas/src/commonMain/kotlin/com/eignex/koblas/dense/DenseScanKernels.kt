package com.eignex.koblas.dense

import kotlin.math.abs

/** Maximum absolute value of a finite array, or `null` when any entry is non-finite. */
internal fun finiteMaxAbs(values: DoubleArray): Double? {
    var maximum = 0.0
    for (value in values) {
        if (!value.isFinite()) return null
        maximum = maxOf(maximum, abs(value))
    }
    return maximum
}

/** Whether [factor] turns any finite array entry into a non-finite product. */
internal fun finiteScaleOverflows(factor: Double, values: DoubleArray): Boolean {
    if (!factor.isFinite() || factor in -1.0..1.0) return false
    for (value in values) if (value.isFinite() && !(factor * value).isFinite()) return true
    return false
}

/** Whether any entry is non-finite. Kept as a numerical leaf for operation-specific eligibility policies. */
internal fun hasNonFinite(values: DoubleArray): Boolean {
    for (value in values) if (!value.isFinite()) return true
    return false
}

/**
 * Maximum magnitude in one selected triangle after scaling, or `null` when a scaled entry is non-finite.
 * A zero factor does not read the triangle, matching BLAS overwrite/no-read decisions made by callers.
 */
internal fun finiteScaledTriangleMaxAbs(values: DoubleArray, n: Int, lower: Boolean, factor: Double): Double? {
    if (factor == 0.0) return 0.0
    var maximum = 0.0
    repeat(n) { column ->
        val from = if (lower) column else 0
        val until = if (lower) n else column + 1
        var row = from
        while (row < until) {
            val scaled = factor * values[row + column * n]
            if (!scaled.isFinite()) return null
            maximum = maxOf(maximum, abs(scaled))
            row++
        }
    }
    return maximum
}

/** Maximum magnitude in one selected finite triangle, or `null` on a non-finite entry. */
internal fun finiteTriangleMaxAbs(values: DoubleArray, n: Int, lower: Boolean): Double? {
    var maximum = 0.0
    repeat(n) { column ->
        val from = if (lower) column else 0
        val until = if (lower) n else column + 1
        var row = from
        while (row < until) {
            val value = values[row + column * n]
            if (!value.isFinite()) return null
            maximum = maxOf(maximum, abs(value))
            row++
        }
    }
    return maximum
}

/** Compares three non-negative finite factors with [limit] without overflowing the comparison. */
internal fun productExceedsBound(limit: Double, first: Double, second: Double, third: Double): Boolean {
    if (first == 0.0 || second == 0.0 || third == 0.0) return false
    var largest = first
    var middle = second
    var smallest = third
    if (largest < middle) largest = middle.also { middle = largest }
    if (middle < smallest) middle = smallest.also { smallest = middle }
    if (largest < middle) largest = middle.also { middle = largest }
    return limit / largest / middle / smallest < 1.0
}

/** Finite maximum magnitude of the selected triangle, treating a unit diagonal as exact one. */
internal fun triangularMultiplyMaxAbs(triangle: DoubleArray, n: Int, lower: Boolean, unitDiagonal: Boolean): Double? {
    var maximum = if (unitDiagonal) 1.0 else 0.0
    repeat(n) { column ->
        val from = if (lower) column else 0
        val until = if (lower) n else column + 1
        var row = from
        while (row < until) {
            if (!unitDiagonal || row != column) {
                val value = triangle[row + column * n]
                if (!value.isFinite()) return null
                maximum = maxOf(maximum, abs(value))
            }
            row++
        }
    }
    return maximum
}

/** Whether one selected triangular block stores an exact zero, excluding no diagonal entries. */
internal fun triangleHasZero(triangle: DoubleArray, offset: Int, size: Int, lda: Int, lower: Boolean): Boolean {
    repeat(size) { column ->
        val from = if (lower) column else 0
        val until = if (lower) size else column + 1
        val base = offset + column * lda
        var row = from
        while (row < until) {
            if (triangle[base + row] == 0.0) return true
            row++
        }
    }
    return false
}
