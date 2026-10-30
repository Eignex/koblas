@file:Suppress("LongParameterList") // BLAS operands, flags and workspace

package com.eignex.koblas.internal.operands

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.dense.applyBeta
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/** Snapshots a region only when it shares the destination buffer. */
internal inline fun <T> stagedRegion(
    workspace: Workspace?,
    a: DenseMatrixRegion,
    destination: DoubleArray,
    stable: DenseMatrixRegion? = null,
    original: DenseMatrixRegion? = null,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrixRegion) -> T,
): T {
    if (a.values !== destination) return block(a)
    if (original != null && stable != null && a.values === original.values && a.offset == original.offset &&
        a.rows == original.rows && a.cols == original.cols && a.leadingDimension == original.leadingDimension
    ) {
        return block(stable)
    }
    return workspace.borrow(DenseMatrix.entryCount(a.rows, a.cols)) { copy ->
        for (j in 0 until a.cols) {
            for (i in 0 until a.rows) {
                if (lower == null || (if (lower) i >= j else i <= j)) {
                    copy[i + j * a.rows] = if (unitDiag && i == j) {
                        1.0
                    } else {
                        a.values[a.offset + i + j * a.leadingDimension]
                    }
                }
            }
        }
        block(DenseMatrix.wrap(a.rows, a.cols, copy))
    }
}

/** Whether two logical rectangles touch, excluding the gaps between their columns. */
internal fun regionsOverlap(a: DenseMatrixRegion, b: DenseMatrixRegion): Boolean {
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

/** Length passed to each scaling kernel, either the full contiguous region or one padded column. */
internal fun scalingLength(rows: Int, columns: Int, leadingDimension: Int): Int = when {
    rows == 0 || columns == 0 -> 0
    leadingDimension == rows -> rows * columns
    else -> rows
}

/** Disjoint submatrices of the same parent need no snapshot. */
internal inline fun <T> stagedRegion(
    workspace: Workspace?,
    a: DenseMatrixRegion,
    destination: DenseMatrixRegion,
    stable: DenseMatrixRegion? = null,
    original: DenseMatrixRegion? = null,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrixRegion) -> T,
): T {
    if (!regionsOverlap(a, destination)) return block(a)
    return stagedRegion(workspace, a, destination.values, stable, original, lower, unitDiag, block)
}

/** Scales selected entries, never reading a zero-scaled destination or visiting its padding. */
internal fun scaleRegion(
    a: DenseMatrixRegion,
    beta: Double,
    lower: Boolean? = null,
    vectors: DenseVectorKernels? = null,
) {
    if (beta == 1.0 || a.rows == 0 || a.cols == 0) return
    if (lower == null && vectors != null) {
        val length = scalingLength(a.rows, a.cols, a.leadingDimension)
        if (a.leadingDimension == a.rows) {
            applyBeta(vectors, a.values, a.offset, length, beta)
        } else {
            for (j in 0 until a.cols) applyBeta(vectors, a.values, a.offset + j * a.leadingDimension, length, beta)
        }
        return
    }
    for (j in 0 until a.cols) {
        val first = if (lower == true) j else 0
        val last = if (lower == false) j + 1 else a.rows
        for (i in first until last) a[i, j] = if (beta == 0.0) 0.0 else beta * a[i, j]
    }
}

/** Exact overlap of a vector run and logical matrix columns, ignoring column padding. */
internal fun regionVectorOverlap(a: DenseMatrixRegion, x: DenseVector): Boolean {
    if (a.values !== x.values || x.size == 0 || a.rows == 0 || a.cols == 0) return false
    val step = if (x.size <= 1) 1L else kotlin.math.abs(x.stride.toLong())
    val first = minOf(x.offset.toLong(), x.offset.toLong() + (x.size - 1L) * x.stride)
    val last = maxOf(x.offset.toLong(), x.offset.toLong() + (x.size - 1L) * x.stride)
    for (column in 0 until a.cols) {
        val from = a.offset.toLong() + column.toLong() * a.leadingDimension
        val until = from + a.rows
        val candidate = first + maxOf(0L, (from - first + step - 1) / step) * step
        if (candidate <= last && candidate < until) return true
    }
    return false
}

/** A vector destination snapshots only overlapping logical matrix entries. */
internal inline fun <T> stagedRegion(
    workspace: Workspace?,
    a: DenseMatrixRegion,
    destination: DenseVector,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrixRegion) -> T,
): T {
    if (!regionVectorOverlap(a, destination)) return block(a)
    return stagedRegion(workspace, a, destination.values, lower = lower, unitDiag = unitDiag, block = block)
}

/** Whether two regions have identical logical-to-physical addressing. */
internal fun sameRegion(a: DenseMatrixRegion, b: DenseMatrixRegion): Boolean =
    a.values === b.values && a.offset == b.offset && a.rows == b.rows && a.cols == b.cols &&
        a.leadingDimension == b.leadingDimension
