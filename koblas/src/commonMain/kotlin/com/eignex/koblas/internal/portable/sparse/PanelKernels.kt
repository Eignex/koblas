@file:Suppress("LongParameterList", "TooManyFunctions") // Primitive windows carry arrays, offsets, strides and extents.

package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.internal.portable.dense.PANEL_NARROW_GROUP

internal fun sparseColumnUpdate(
    alpha: Double,
    a: DoubleArray, aOffset: Int, rowStride: Int, indexStride: Int,
    indices: IntArray,
    values: DoubleArray, fromIndex: Int, columns: Int, rows: Int,
    y: DoubleArray, yOffset: Int,
) {
    if (rows <= 0 || columns <= 0) return
    var c = 0
    while (c + PANEL_NARROW_GROUP <= columns) {
        val t0 = alpha * values[fromIndex + c]
        val t1 = alpha * values[fromIndex + c + 1]
        val at0 = aOffset + indices[fromIndex + c] * indexStride
        val at1 = aOffset + indices[fromIndex + c + 1] * indexStride
        var from = 0
        for (i in 0 until rows) {
            y[yOffset + i] += t0 * a[at0 + from] + t1 * a[at1 + from]
            from += rowStride
        }
        c += PANEL_NARROW_GROUP
    }
    while (c < columns) {
        val t = alpha * values[fromIndex + c]
        val at = aOffset + indices[fromIndex + c] * indexStride
        var from = 0
        for (i in 0 until rows) {
            y[yOffset + i] += t * a[at + from]
            from += rowStride
        }
        c++
    }
}

internal fun sparseRankUpdate(
    alpha: Double,
    a: DoubleArray, aOffset: Int, rowStride: Int, indexStride: Int,
    indices: IntArray,
    values: DoubleArray, fromIndex: Int, columns: Int, rows: Int,
    x: DoubleArray, xOffset: Int,
) {
    if (rows <= 0 || columns <= 0) return
    var c = 0
    while (c + PANEL_NARROW_GROUP <= columns) {
        val t0 = alpha * values[fromIndex + c]
        val t1 = alpha * values[fromIndex + c + 1]
        val at0 = aOffset + indices[fromIndex + c] * indexStride
        val at1 = aOffset + indices[fromIndex + c + 1] * indexStride
        var to = 0
        for (i in 0 until rows) {
            val v = x[xOffset + i]
            a[at0 + to] += t0 * v
            a[at1 + to] += t1 * v
            to += rowStride
        }
        c += PANEL_NARROW_GROUP
    }
    while (c < columns) {
        val t = alpha * values[fromIndex + c]
        val at = aOffset + indices[fromIndex + c] * indexStride
        var to = 0
        for (i in 0 until rows) {
            a[at + to] += t * x[xOffset + i]
            to += rowStride
        }
        c++
    }
}

internal fun sparseCoupledUpdate(
    alpha: Double,
    a: DoubleArray,
    b: DoubleArray, offset: Int, rowStride: Int, indexStride: Int,
    indices: IntArray,
    values: DoubleArray, fromIndex: Int, columns: Int, rows: Int, pivot: Int,
    sums: DoubleArray, sumOffset: Int, excluded: Int, destinationOffset: Int, destinationRowStride: Int,
    destinationIndexStride: Int,
) {
    if (rows <= 0 || columns <= 0) return
    for (c in 0 until columns) {
        val t = alpha * values[fromIndex + c]
        val at = offset + indices[fromIndex + c] * indexStride
        var position = 0
        if (c == excluded) {
            for (i in 0 until rows) {
                a[destinationOffset + indices[fromIndex + c] * destinationIndexStride + i * destinationRowStride] +=
                    t * b[pivot + position]
                position += rowStride
            }
        } else {
            for (i in 0 until rows) {
                a[destinationOffset + indices[fromIndex + c] * destinationIndexStride + i * destinationRowStride] +=
                    t * b[pivot + position]
                sums[sumOffset + i * destinationRowStride] += t * b[at + position]
                position += rowStride
            }
        }
    }
}

/** Sparse column indices are reused across this many right-hand sides. */
internal const val SPARSE_RHS_GROUP = 8

/** A strided reduction keeps its single accumulator in a local. */
internal const val SPARSE_REDUCTION_GROUP = 1
