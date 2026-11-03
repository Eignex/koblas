@file:Suppress("LongParameterList") // two addressed windows, each an array, an origin, a length and a stride

package com.eignex.koblas.internal.operands

import com.eignex.koblas.*
import com.eignex.koblas.engine.*

/*
 * Overlap tests and snapshots over vector windows given as an array, an origin, a length and a stride, so a
 * routine that receives its vectors that way never builds a vector object to stage them.
 */

/** Exact overlap of a vector window and the logical columns of [a], ignoring column padding. */
internal fun regionWindowOverlap(
    a: DenseMatrixRegion,
    values: DoubleArray,
    offset: Int,
    count: Int,
    stride: Int,
): Boolean {
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

/** [block] over the window, or over a contiguous copy of its entries when [aliased]. */
internal inline fun <T> stagedWindow(
    arrays: ArraySupplier?,
    values: DoubleArray,
    offset: Int,
    count: Int,
    stride: Int,
    aliased: Boolean,
    block: (DoubleArray, Int, Int) -> T,
): T {
    if (!aliased) return block(values, offset, stride)
    return arrays.borrow(count) { copy ->
        for (i in 0 until count) copy[i] = values[offset + i * stride]
        block(copy, 0, 1)
    }
}

/** [block] over [a], or over a snapshot of it when its logical entries overlap the destination window. */
internal inline fun <T> stagedRegionAgainst(
    arrays: ArraySupplier?,
    a: DenseMatrixRegion,
    values: DoubleArray,
    offset: Int,
    count: Int,
    stride: Int,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrixRegion) -> T,
): T {
    if (!regionWindowOverlap(a, values, offset, count, stride)) return block(a)
    return stagedRegion(arrays, a, values, lower = lower, unitDiag = unitDiag, block = block)
}

/** Stages two vector windows against a matrix destination, sharing the snapshot of identical windows. */
internal inline fun <T> stagedWindows(
    arrays: ArraySupplier?,
    x: DoubleArray,
    xOffset: Int,
    xCount: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yCount: Int,
    yStride: Int,
    destination: DenseMatrixRegion,
    block: (DoubleArray, Int, Int, DoubleArray, Int, Int) -> T,
): T {
    val xAliased = regionWindowOverlap(destination, x, xOffset, xCount, xStride)
    return stagedWindow(arrays, x, xOffset, xCount, xStride, xAliased) { sx, sxo, sxs ->
        val same = x === y && xOffset == yOffset && xCount == yCount && (xCount <= 1 || xStride == yStride)
        if (same) {
            block(sx, sxo, sxs, sx, sxo, sxs)
        } else {
            val yAliased = regionWindowOverlap(destination, y, yOffset, yCount, yStride)
            stagedWindow(arrays, y, yOffset, yCount, yStride, yAliased) { sy, syo, sys ->
                block(sx, sxo, sxs, sy, syo, sys)
            }
        }
    }
}
