package com.eignex.koblas.internal.sparse

import com.eignex.koblas.internal.numeric.MIN_NORMAL
import com.eignex.koblas.internal.numeric.rescaledEuclideanNorm
import com.eignex.koblas.sparse.SparseOperation
import kotlin.math.sqrt

/** Trusted indexed windows, with portable defaults for every unselected operation. */
internal interface IndexedSparseKernels {
    val name: String
    fun implementationFor(operation: SparseOperation, count: Int): String? = name

    fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double {
        var sum = 0.0
        for (k in 0 until count) sum += values[valueOffset + k] * dense[indices[indexOffset + k]]
        return sum
    }

    fun dotSparse(
        xIndices: IntArray,
        xIndexOffset: Int,
        xValues: DoubleArray,
        xValueOffset: Int,
        xCount: Int,
        yIndices: IntArray,
        yIndexOffset: Int,
        yValues: DoubleArray,
        yValueOffset: Int,
        yCount: Int,
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

    fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ) {
        for (k in 0 until count) destination[indices[indexOffset + k]] += alpha * values[valueOffset + k]
    }

    fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        for (k in 0 until count) destination[indices[indexOffset + k]] = values[valueOffset + k]
    }

    fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        for (k in 0 until count) values[valueOffset + k] = source[indices[indexOffset + k]]
    }

    fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        for (k in 0 until count) {
            val index = indices[indexOffset + k]
            values[valueOffset + k] = source[index]
            source[index] = 0.0
        }
    }

    fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
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

    private fun indexedSquareSum(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
        // Independent sums avoid serialising every indexed load behind one addition chain.
        var squares = 0.0
        var squares1 = 0.0
        var squares2 = 0.0
        var squares3 = 0.0
        var k = 0
        val bound = count - 3
        while (k < bound) {
            val value = values[indices[indexOffset + k]]
            val value1 = values[indices[indexOffset + k + 1]]
            val value2 = values[indices[indexOffset + k + 2]]
            val value3 = values[indices[indexOffset + k + 3]]
            squares += value * value
            squares1 += value1 * value1
            squares2 += value2 * value2
            squares3 += value3 * value3
            k += 4
        }
        squares += squares1 + squares2 + squares3
        while (k < count) {
            val value = values[indices[indexOffset + k]]
            squares += value * value
            k++
        }
        return squares
    }
}

// Window conveniences dispatch through the selected override, including delegated implementations.
internal fun IndexedSparseKernels.gather(indices: IntArray, values: DoubleArray, source: DoubleArray) =
    gather(indices, 0, values, 0, values.size, source)

internal fun IndexedSparseKernels.dotDense(indices: IntArray, values: DoubleArray, dense: DoubleArray): Double =
    dotDense(indices, 0, values, 0, values.size, dense)

internal fun IndexedSparseKernels.dotSparse(
    xIndices: IntArray,
    xValues: DoubleArray,
    yIndices: IntArray,
    yValues: DoubleArray,
): Double = dotSparse(xIndices, 0, xValues, 0, xValues.size, yIndices, 0, yValues, 0, yValues.size)

internal fun IndexedSparseKernels.axpy(
    indices: IntArray,
    values: DoubleArray,
    alpha: Double,
    destination: DoubleArray,
) = axpy(indices, 0, values, 0, values.size, alpha, destination)

internal fun IndexedSparseKernels.scatter(indices: IntArray, values: DoubleArray, destination: DoubleArray) =
    scatter(indices, 0, values, 0, values.size, destination)

internal fun IndexedSparseKernels.gatherZero(indices: IntArray, values: DoubleArray, source: DoubleArray) =
    gatherZero(indices, 0, values, 0, values.size, source)
