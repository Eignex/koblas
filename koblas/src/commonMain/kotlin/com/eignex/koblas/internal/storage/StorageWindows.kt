@file:Suppress("LongParameterList") // Physical windows have independent offsets, sizes and strides.

package com.eignex.koblas.internal.storage

import com.eignex.koblas.*

/** Whether the logical matrix occupies the whole buffer in compact column-major order. */
internal val DenseMatrix.isWholeBuffer: Boolean
    get() = offset == 0 && leadingDimension == rows && values.size.toLong() == rows.toLong() * cols

/** Whether this vector spans its entire backing array in order. */
internal val DenseVector.isWholeArray: Boolean
    get() = offset == 0 && stride == 1 && values.size == size

/** Exact overlap of a vector window and the logical columns of [a], ignoring column padding. */
internal fun regionWindowOverlap(a: DenseMatrix, values: DoubleArray, offset: Int, count: Int, stride: Int): Boolean {
    if (a.values !== values || count == 0 || a.rows == 0 || a.cols == 0) return false
    val step = if (count <= 1) 1L else kotlin.math.abs(stride.toLong())
    val first = minOf(offset.toLong(), offset.toLong() + (count - 1L) * stride)
    val last = maxOf(offset.toLong(), offset.toLong() + (count - 1L) * stride)
    for (column in 0 until a.cols) {
        val from = a.offset.toLong() + column.toLong() * a.leadingDimension
        val until = from + a.rows
        val candidate = first + maxOf(0L, (from - first + step - 1) / step) * step
        if (candidate <= last && candidate < until) return true
    }
    return false
}

/** Whether two vector windows address at least one common buffer entry. */
internal fun windowsOverlap(
    aValues: DoubleArray,
    aOffset: Int,
    aCount: Int,
    aStride: Int,
    bValues: DoubleArray,
    bOffset: Int,
    bCount: Int,
    bStride: Int,
): Boolean {
    if (aValues !== bValues || aCount == 0 || bCount == 0) return false
    val aLast = aOffset.toLong() + (aCount - 1).toLong() * aStride
    val bLast = bOffset.toLong() + (bCount - 1).toLong() * bStride
    val low = maxOf(minOf(aOffset.toLong(), aLast), minOf(bOffset.toLong(), bLast))
    val high = minOf(maxOf(aOffset.toLong(), aLast), maxOf(bOffset.toLong(), bLast))
    if (low > high) return false
    // For equal spacings the intersecting bounds contain a shared entry exactly when the two origins belong to
    // the same index lattice.
    val step = kotlin.math.abs(aStride.toLong())
    if (step == kotlin.math.abs(bStride.toLong())) return (aOffset.toLong() - bOffset) % step == 0L
    val aShorter = aCount <= bCount
    val firstOffset = if (aShorter) aOffset else bOffset
    val firstCount = if (aShorter) aCount else bCount
    val firstStride = if (aShorter) aStride else bStride
    val secondOffset = if (aShorter) bOffset else aOffset
    val secondCount = if (aShorter) bCount else aCount
    val secondStride = if (aShorter) bStride else aStride
    for (i in 0 until firstCount) {
        val relative = firstOffset + i * firstStride - secondOffset
        if (relative % secondStride == 0 && relative / secondStride in 0 until secondCount) return true
    }
    return false
}

/** Whether two logical rectangles touch, excluding the gaps between their columns. */
internal fun regionsOverlap(a: DenseMatrix, b: DenseMatrix): Boolean {
    if (a.values !== b.values || a.rows == 0 || a.cols == 0 || b.rows == 0 || b.cols == 0) return false
    var left = 0
    var right = 0
    while (left < a.cols && right < b.cols) {
        val aStart = a.offset + left * a.leadingDimension
        val bStart = b.offset + right * b.leadingDimension
        if (aStart < bStart + b.rows && bStart < aStart + a.rows) return true
        if (aStart < bStart) left++ else right++
    }
    return false
}

/** Whether two regions have identical logical-to-physical addressing. */
internal fun sameRegion(a: DenseMatrix, b: DenseMatrix): Boolean =
    a.values === b.values && a.offset == b.offset && a.rows == b.rows && a.cols == b.cols &&
        a.leadingDimension == b.leadingDimension

/** Identical logical vector addressing can share one immutable snapshot. */
internal fun sameVectorMapping(a: DenseVector, b: DenseVector): Boolean =
    a.values === b.values && a.offset == b.offset && a.size == b.size && (a.size <= 1 || a.stride == b.stride)

/** Whether these vectors address at least one common buffer entry. */
internal fun DenseVector.overlaps(other: DenseVector): Boolean =
    windowsOverlap(values, offset, size, stride, other.values, other.offset, other.size, other.stride)
