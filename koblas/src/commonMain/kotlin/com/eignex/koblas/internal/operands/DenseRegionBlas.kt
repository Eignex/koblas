@file:Suppress("LongParameterList") // BLAS operands, flags and scratch

package com.eignex.koblas.internal.operands

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.dense.applyBeta

/** Snapshots a region only when it shares the destination buffer. */
internal inline fun <T> stagedRegion(
    arrays: Scratch?,
    a: DenseMatrixRegion,
    destination: DoubleArray,
    stable: DenseMatrixRegion? = null,
    original: DenseMatrixRegion? = null,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrixRegion) -> T,
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

/** Disjoint submatrices of the same parent need no snapshot. */
internal inline fun <T> stagedRegion(
    arrays: Scratch?,
    a: DenseMatrixRegion,
    destination: DenseMatrixRegion,
    stable: DenseMatrixRegion? = null,
    original: DenseMatrixRegion? = null,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrixRegion) -> T,
): T {
    if (!regionsOverlap(a, destination)) return block(a)
    return stagedRegion(arrays, a, destination.values, stable, original, lower, unitDiag, block)
}

/** Scales selected entries, never reading a zero-scaled destination or visiting its padding. */
internal fun scaleRegion(a: DenseMatrixRegion, beta: Double, lower: Boolean? = null, vectors: KoblasEngine? = null) {
    if (beta == 1.0 || a.rows == 0 || a.cols == 0) return
    if (lower == null && vectors != null) {
        if (a.leadingDimension == a.rows) {
            applyBeta(vectors, a.values, a.offset, a.rows * a.cols, beta)
        } else {
            for (j in 0 until a.cols) applyBeta(vectors, a.values, a.offset + j * a.leadingDimension, a.rows, beta)
        }
        return
    }
    for (j in 0 until a.cols) {
        val first = if (lower == true) j else 0
        val last = if (lower == false) j + 1 else a.rows
        for (i in first until last) a[i, j] = if (beta == 0.0) 0.0 else beta * a[i, j]
    }
}

/** Whether two regions have identical logical-to-physical addressing. */
internal fun sameRegion(a: DenseMatrixRegion, b: DenseMatrixRegion): Boolean =
    a.values === b.values && a.offset == b.offset && a.rows == b.rows && a.cols == b.cols &&
        a.leadingDimension == b.leadingDimension
