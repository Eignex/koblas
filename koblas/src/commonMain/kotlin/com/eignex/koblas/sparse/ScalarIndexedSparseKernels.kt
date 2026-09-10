package com.eignex.koblas.sparse

import com.eignex.koblas.internal.numeric.MIN_NORMAL
import kotlin.math.abs
import kotlin.math.sqrt

/** Scalar indexed kernels used as the semantic oracle and universal fallback. */
internal object ScalarIndexedSparseKernels : IndexedSparseKernels {
    override fun dotDense(
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

    override fun dotSparse(
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

    override fun axpy(
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

    override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        for (k in 0 until count) destination[indices[indexOffset + k]] = values[valueOffset + k]
    }

    override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        for (k in 0 until count) values[valueOffset + k] = source[indices[indexOffset + k]]
    }

    override fun gatherZero(
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

    override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
        var squares = 0.0
        for (k in 0 until count) {
            val value = values[indices[indexOffset + k]]
            squares += value * value
        }
        if (squares.isFinite() && squares >= MIN_NORMAL) return sqrt(squares)

        var maximum = 0.0
        for (k in 0 until count) {
            val magnitude = abs(values[indices[indexOffset + k]])
            if (magnitude > maximum) maximum = magnitude
        }
        if (maximum == 0.0 || maximum.isInfinite()) return sqrt(squares)

        var scaledSquares = 0.0
        for (k in 0 until count) {
            val scaled = values[indices[indexOffset + k]] / maximum
            scaledSquares += scaled * scaled
        }
        return maximum * sqrt(scaledSquares)
    }
}
