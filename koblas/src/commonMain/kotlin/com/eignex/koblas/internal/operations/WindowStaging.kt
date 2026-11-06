@file:Suppress("LongParameterList") // two addressed windows, each an array, an origin, a length and a stride

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.regionWindowOverlap

/*
 * Overlap tests and snapshots over vector windows given as an array, an origin, a length and a stride, so a
 * routine that receives its vectors that way never builds a vector object to stage them.
 */

/** [block] over the window, or over a contiguous copy of its entries when [aliased]. */
internal inline fun <T> stagedWindow(
    arrays: Scratch?,
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
    arrays: Scratch?,
    a: DenseMatrix,
    values: DoubleArray,
    offset: Int,
    count: Int,
    stride: Int,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrix) -> T,
): T {
    if (!regionWindowOverlap(a, values, offset, count, stride)) return block(a)
    return stagedRegion(arrays, a, values, lower = lower, unitDiag = unitDiag, block = block)
}

/** Stages two vector windows against a matrix destination, sharing the snapshot of identical windows. */
internal inline fun <T> stagedWindows(
    arrays: Scratch?,
    x: DoubleArray,
    xOffset: Int,
    xCount: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yCount: Int,
    yStride: Int,
    destination: DenseMatrix,
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
