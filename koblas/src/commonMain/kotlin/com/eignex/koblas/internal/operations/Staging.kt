@file:Suppress("LongParameterList") // Independent windows and extents.

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.*
import com.eignex.koblas.internal.storage.dense.*

/* Alias snapshots use arrays loans; callers decide which operands need staging.
 * Sparse snapshots share read-only structure and copy only values.
 */

/** [block] over [values], or over a copy of it when [aliased]. */
internal inline fun <T> staged(arrays: Scratch?, values: DoubleArray, aliased: Boolean, block: (DoubleArray) -> T): T {
    if (!aliased) return block(values)
    return arrays.borrow(values.size) { copy ->
        values.copyInto(copy)
        block(copy)
    }
}

/** Equal mappings have no cross-index dependency; partial aliases need a snapshot. */
internal fun DenseVector.stableFor(destination: DenseVector): DenseVector =
    if (!sameVectorMapping(this, destination) && overlaps(destination)) DenseVector.wrap(toDoubleArray()) else this

/** Opaque storage may read the destination without exposing its buffer. */
internal fun Vector.stableFor(destination: DenseVector): Vector = when (this) {
    is SparseVector -> if (values === destination.values) copyVectorStorage(this) else this
    is DenseVector -> stableFor(destination)
    else -> DenseVector.wrap(toDoubleArray())
}

/** A matrix snapshot keeps its geometry in primitive fields rather than a temporary container. */
internal inline fun <T> stagedRegion(
    arrays: Scratch?,
    a: DenseMatrix,
    destination: DoubleArray,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DoubleArray, Int, Int) -> T,
): T {
    if (a.values !== destination) return block(a.values, a.offset, a.leadingDimension)
    return arrays.borrow(DenseMatrix.entryCount(a.rows, a.cols)) { copy ->
        val values = a.values
        val offset = a.offset
        val lda = a.leadingDimension
        copyTriangle(a.rows, a.cols, lower, unitDiag, copy) { i, j -> values[offset + i + j * lda] }
        block(copy, 0, a.rows)
    }
}

/** Primitive snapshot geometry for exact rectangle overlap, excluding the parent's padding. */
internal inline fun <T> stagedRegion(
    arrays: Scratch?,
    a: DenseMatrix,
    destination: DenseMatrix,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DoubleArray, Int, Int) -> T,
): T {
    if (!regionsOverlap(a, destination)) return block(a.values, a.offset, a.leadingDimension)
    return stagedRegion(arrays, a, destination.values, lower, unitDiag, block)
}

/** Snapshot logical entries into a loan without creating a temporary vector descriptor. */
internal inline fun <T> stagedWindow(
    arrays: Scratch?,
    values: DoubleArray,
    offset: Int,
    size: Int,
    stride: Int,
    aliased: Boolean,
    block: (DoubleArray, Int, Int) -> T,
): T {
    if (!aliased) return block(values, offset, stride)
    return arrays.borrow(size) { copy ->
        for (i in 0 until size) copy[i] = values[offset + i * stride]
        block(copy, 0, 1)
    }
}

/** Matrix snapshots against a vector preserve primitive geometry for every backend. */
internal inline fun <T> stagedRegionAgainst(
    arrays: Scratch?,
    a: DenseMatrix,
    vector: DenseVector,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DoubleArray, Int, Int) -> T,
): T {
    if (!regionWindowOverlap(a, vector.values, vector.offset, vector.size, vector.stride)) {
        return block(a.values, a.offset, a.leadingDimension)
    }
    return stagedRegion(arrays, a, vector.values, lower, unitDiag, block)
}

/** Primitive window pair with one loan for identical mappings, even for single entries with different strides. */
internal inline fun <T> stagedWindows(
    arrays: Scratch?,
    x: DoubleArray,
    xOffset: Int,
    xSize: Int,
    xStride: Int,
    xAliased: Boolean,
    y: DoubleArray,
    yOffset: Int,
    ySize: Int,
    yStride: Int,
    yAliased: Boolean,
    block: (DoubleArray, Int, Int, DoubleArray, Int, Int) -> T,
): T = stagedWindow(arrays, x, xOffset, xSize, xStride, xAliased) { xv, xo, xs ->
    if (x === y && xOffset == yOffset && xSize == ySize && (xSize <= 1 || xStride == yStride)) {
        block(xv, xo, xs, xv, xo, xs)
    } else {
        stagedWindow(arrays, y, yOffset, ySize, yStride, yAliased) { yv, yo, ys -> block(xv, xo, xs, yv, yo, ys) }
    }
}

/**
 * Writes the [lower] or upper triangle of a [rows] by [columns] operand, or all of it when [lower] is null,
 * into [copy] with [rows] as its leading dimension. A [unitDiag] diagonal is written as 1 without calling
 * [entry]; the other triangle is neither read nor written.
 */
internal inline fun copyTriangle(
    rows: Int,
    columns: Int,
    lower: Boolean?,
    unitDiag: Boolean,
    copy: DoubleArray,
    entry: (i: Int, j: Int) -> Double,
) {
    for (j in 0 until columns) {
        val first = if (lower == true) j else 0
        val last = if (lower == false && j < rows) j + 1 else rows
        for (i in first until last) copy[i + j * rows] = if (unitDiag && i == j) 1.0 else entry(i, j)
    }
}
