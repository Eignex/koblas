@file:Suppress("LongParameterList", "TooManyFunctions") // Primitive windows carry arrays, offsets, strides and extents.

package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.internal.portable.dense.MIN_NORMAL
import com.eignex.koblas.internal.portable.dense.rescaledEuclideanNorm
import kotlin.math.sqrt

internal fun indexedDotDense(
    indices: IntArray, indexOffset: Int,
    values: DoubleArray, valueOffset: Int, count: Int,
    dense: DoubleArray,
): Double {
    var sum = 0.0
    for (k in 0 until count) sum += values[valueOffset + k] * dense[indices[indexOffset + k]]
    return sum
}

internal fun indexedDotSparse(
    xIndices: IntArray, xIndexOffset: Int,
    xValues: DoubleArray, xValueOffset: Int, xCount: Int,
    yIndices: IntArray, yIndexOffset: Int,
    yValues: DoubleArray, yValueOffset: Int, yCount: Int,
): Double {
    var sum = 0.0
    var a = 0
    var b = 0
    while (a < xCount && b < yCount) {
        val ia = xIndices[xIndexOffset + a]
        val ib = yIndices[yIndexOffset + b]
        when {
            ia < ib -> a++

            ia > ib -> b++

            else -> {
                sum += xValues[xValueOffset + a] * yValues[yValueOffset + b]
                a++
                b++
            }
        }
    }
    return sum
}

internal fun indexedAxpy(
    indices: IntArray, indexOffset: Int,
    values: DoubleArray, valueOffset: Int, count: Int, alpha: Double,
    destination: DoubleArray,
) {
    for (k in 0 until count) destination[indices[indexOffset + k]] += alpha * values[valueOffset + k]
}

internal fun indexedScatter(
    indices: IntArray, indexOffset: Int,
    values: DoubleArray, valueOffset: Int, count: Int,
    destination: DoubleArray,
) {
    for (k in 0 until count) destination[indices[indexOffset + k]] = values[valueOffset + k]
}

internal fun indexedGather(
    indices: IntArray, indexOffset: Int,
    values: DoubleArray, valueOffset: Int, count: Int,
    source: DoubleArray,
) {
    for (k in 0 until count) values[valueOffset + k] = source[indices[indexOffset + k]]
}

internal fun indexedGatherZero(
    indices: IntArray, indexOffset: Int,
    values: DoubleArray, valueOffset: Int, count: Int,
    source: DoubleArray,
) {
    for (k in 0 until count) {
        val index = indices[indexOffset + k]
        values[valueOffset + k] = source[index]
        source[index] = 0.0
    }
}

internal fun indexedNrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
    var squares = 0.0
    if (count >= 64) {
        squares = indexedSquareSum(indices, indexOffset, count, values)
    } else {
        for (k in 0 until count) {
            val value = values[indices[indexOffset + k]]
            squares += value * value
        }
    }
    if (squares.isFinite() && squares >= MIN_NORMAL) return sqrt(squares)
    return rescaledEuclideanNorm(count, squares) { values[indices[indexOffset + it]] }
}
