@file:Suppress("LongParameterList", "TooManyFunctions") // Independent windows and extents.

package com.eignex.koblas.internal.kernels.scalar.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.Vector
import com.eignex.koblas.internal.storage.sparse.SparseEntries

internal fun indexedDotDense(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    dense: DoubleArray,
    denseOffset: Int = 0,
    denseStride: Int = 1,
): Double {
    var sum = 0.0
    for (k in 0 until count) {
        val index = denseOffset + indices[indexOffset + k] * denseStride
        sum += values[valueOffset + k] * dense[index]
    }
    return sum
}

internal fun indexedDotSparse(
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

internal fun indexedAxpy(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    alpha: Double,
    destination: DoubleArray,
    destinationOffset: Int = 0,
    destinationStride: Int = 1,
) {
    for (k in 0 until count) {
        val index = destinationOffset + indices[indexOffset + k] * destinationStride
        destination[index] += alpha * values[valueOffset + k]
    }
}

internal fun indexedScatter(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    destination: DoubleArray,
    destinationOffset: Int = 0,
    destinationStride: Int = 1,
) {
    for (k in 0 until count) {
        val index = destinationOffset + indices[indexOffset + k] * destinationStride
        destination[index] = values[valueOffset + k]
    }
}

internal fun indexedGather(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    source: DoubleArray,
) {
    for (k in 0 until count) values[valueOffset + k] = source[indices[indexOffset + k]]
}

internal fun indexedGatherZero(
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

/** Merge sparse supports without scanning the logical domain or evaluating absent products. */
internal fun scalarSparseVectorAxpy(alpha: Double, x: Vector, y: SparseVector): SparseVector {
    val entries = SparseEntries()
    var b = 0
    if (x is SparseVector) {
        var a = 0
        while (a < x.nnz || b < y.nnz) {
            val ia = if (a < x.nnz) x.indices[x.offset + a] else Int.MAX_VALUE
            val ib = if (b < y.nnz) y.indices[y.offset + b] else Int.MAX_VALUE
            val index = minOf(ia, ib)
            val previous = if (ib == index) y.values[y.offset + b++] else 0.0
            val value = if (ia == index) previous + alpha * x.values[x.offset + a++] else previous
            entries.add(index, value)
        }
    } else {
        for (i in 0 until x.size) {
            val stored = b < y.nnz && y.indices[y.offset + b] == i
            val previous = if (stored) y.values[y.offset + b++] else 0.0
            val value = previous + alpha * x[i]
            if (stored || value.toBits() != 0L) {
                entries.add(i, value)
            }
        }
    }
    return entries.toVector(y.size)
}
