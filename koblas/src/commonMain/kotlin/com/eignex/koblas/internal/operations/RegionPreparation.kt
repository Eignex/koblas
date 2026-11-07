@file:Suppress("LongParameterList") // BLAS operands, flags and scratch

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.storage.*

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

/** Scales selected entries, never reading a zero-scaled destination or visiting its padding. */
internal fun scaleRegion(a: DenseMatrix, beta: Double, lower: Boolean? = null, vectors: KoblasEngine? = null) {
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

/** `v = beta * v` over the [len] entries from [off], honoring the `beta == 0` overwrite convention. */
internal fun applyBeta(k: KoblasEngine, v: DoubleArray, off: Int, len: Int, beta: Double, stride: Int = 1) {
    when {
        beta == 0.0 -> if (stride == 1) {
            v.fill(
                0.0,
                off,
                off + len,
            )
        } else {
            for (i in 0 until len) v[off + i * stride] = 0.0
        }

        beta != 1.0 -> k.kernels.scale(v, off, beta, len, stride)
    }
}
