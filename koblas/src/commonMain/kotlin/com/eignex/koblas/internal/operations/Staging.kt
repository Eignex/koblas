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

/**
 * [staged] for a sparse operand, where only the coefficients can alias.
 *
 * The structural arrays are never a destination, so the snapshot shares them and copies the coefficients
 * alone.
 */
@OptIn(UnsafeKoblasApi::class)
internal inline fun <T> staged(arrays: Scratch?, a: SparseMatrix, aliased: Boolean, block: (SparseMatrix) -> T): T {
    if (!aliased) return block(a)
    return arrays.borrow(a.nnz) { copy ->
        a.values.copyInto(copy)
        block(SparseMatrix.wrapTrusted(a.rows, a.cols, a.colPointers, a.rowIndices, copy))
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

/** Snapshots a region only when it shares the destination buffer. */
internal inline fun <T> stagedRegion(
    arrays: Scratch?,
    a: DenseMatrix,
    destination: DoubleArray,
    stable: DenseMatrix? = null,
    original: DenseMatrix? = null,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrix) -> T,
): T {
    if (a.values !== destination) return block(a)
    if (original != null && stable != null && sameRegion(a, original)) return block(stable)
    return arrays.borrow(DenseMatrix.entryCount(a.rows, a.cols)) { copy ->
        val values = a.values
        val offset = a.offset
        val lda = a.leadingDimension
        copyTriangle(a.rows, a.cols, lower, unitDiag, copy) { i, j -> values[offset + i + j * lda] }
        block(DenseMatrix.wrap(a.rows, a.cols, copy))
    }
}

/** Disjoint submatrices of the same parent need no snapshot. */
internal inline fun <T> stagedRegion(
    arrays: Scratch?,
    a: DenseMatrix,
    destination: DenseMatrix,
    stable: DenseMatrix? = null,
    original: DenseMatrix? = null,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrix) -> T,
): T {
    if (!regionsOverlap(a, destination)) return block(a)
    return stagedRegion(arrays, a, destination.values, stable, original, lower, unitDiag, block)
}

/** [block] over [vector], or over a contiguous snapshot of its logical entries when [aliased]. */
internal inline fun <T> stagedWindow(
    arrays: Scratch?,
    vector: DenseVector,
    aliased: Boolean,
    block: (DenseVector) -> T,
): T {
    if (!aliased) return block(vector)
    return arrays.borrow(vector.size) { copy ->
        for (i in 0 until vector.size) copy[i] = vector[i]
        block(DenseVector.wrap(copy))
    }
}

/** Snapshots [a] only when its logical entries overlap [vector]. */
internal inline fun <T> stagedRegionAgainst(
    arrays: Scratch?,
    a: DenseMatrix,
    vector: DenseVector,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrix) -> T,
): T {
    if (!regionWindowOverlap(a, vector.values, vector.offset, vector.size, vector.stride)) return block(a)
    return stagedRegion(arrays, a, vector.values, lower = lower, unitDiag = unitDiag, block = block)
}

/** Stages both vectors against [destination], sharing a snapshot when they address the same entries. */
internal inline fun <T> stagedWindows(
    arrays: Scratch?,
    x: DenseVector,
    y: DenseVector,
    destination: DenseMatrix,
    block: (DenseVector, DenseVector) -> T,
): T {
    val xAliased = regionWindowOverlap(destination, x.values, x.offset, x.size, x.stride)
    return stagedWindow(arrays, x, xAliased) { sx ->
        if (sameVectorMapping(x, y)) {
            block(sx, sx)
        } else {
            val yAliased = regionWindowOverlap(destination, y.values, y.offset, y.size, y.stride)
            stagedWindow(arrays, y, yAliased) { sy -> block(sx, sy) }
        }
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
