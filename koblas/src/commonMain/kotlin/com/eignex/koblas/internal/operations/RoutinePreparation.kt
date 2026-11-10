@file:Suppress("LongParameterList") // Independent windows and extents.

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.scheduling.dense.scheduleGer
import com.eignex.koblas.internal.storage.*
import com.eignex.koblas.internal.storage.dense.*

/*
 * Shared portable and vendor preparation settles empty and zero-multiplier calls before arithmetic,
 * and snapshots overlapping inputs. Nonoverlapping vector views retain their addressing until the kernel.
 */

/**
 * `y = alpha · op(A) · x + beta · y`, for `gemv` and, from one [lower] triangle, `symv`. [body] receives the
 * stable matrix and the stable `x` window.
 */
internal inline fun prepareMatrixVector(
    vectors: KoblasEngine,
    alpha: Double,
    a: DenseMatrix,
    transpose: Boolean,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    lower: Boolean?,
    arrays: Scratch?,
    body: (DenseMatrix, DenseVector) -> Unit,
) {
    val rows = if (transpose) a.cols else a.rows
    val depth = if (transpose) a.rows else a.cols
    requireGemvOperands(a, transpose, x.size, y.size)
    if (rows == 0) return
    if (alpha == 0.0 || depth == 0) {
        applyBeta(vectors, y.values, y.offset, rows, beta, y.stride)
        return
    }
    stagedRegionAgainst(arrays, a, y, lower = lower) { sa ->
        stagedWindow(arrays, x, x.overlaps(y)) { sx -> body(sa, sx) }
    }
}

/**
 * `A = A + alpha · x · yᵀ` and its symmetric forms, with `x` spanning `A`'s rows and `y` its columns. [body]
 * receives the stable windows; identical windows share one snapshot.
 */
internal inline fun prepareRankUpdate(
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrix,
    arrays: Scratch?,
    body: (DenseVector, DenseVector) -> Unit,
) {
    requireGerOperands(x.size, y.size, a)
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
    stagedWindows(arrays, x, y, a, body)
}

/**
 * `op(T) · x = b` or `x = op(T) · x` in place. The substitution overwrites `x` as it goes, so a triangle
 * sharing that storage is snapshotted first.
 */
internal inline fun prepareTriangularVector(
    a: DenseMatrix,
    x: DenseVector,
    lower: Boolean,
    unitDiag: Boolean,
    what: String,
    arrays: Scratch?,
    body: (DenseMatrix) -> Unit,
) {
    requireTriangularVectorOperands(a, x.size, what)
    if (a.rows == 0) return
    stagedRegionAgainst(arrays, a, x, lower = lower, unitDiag = unitDiag, block = body)
}

/**
 * A product into [c] over [depth] shared entries. A zero [alpha] or depth scales [c], or its [selected]
 * triangle, without reading an operand. [aTriangle] limits [a]'s snapshot to the triangle the routine reads.
 */
internal inline fun prepareProduct(
    alpha: Double,
    a: DenseMatrix,
    b: DenseMatrix,
    beta: Double,
    c: DenseMatrix,
    depth: Int,
    selected: Boolean?,
    aTriangle: Boolean?,
    vectors: KoblasEngine?,
    arrays: Scratch?,
    body: (DenseMatrix, DenseMatrix) -> Unit,
) {
    // Return before borrowing operand-sized scratch for an empty destination.
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, selected, vectors)
        return
    }
    stagedRegion(arrays, a, c, lower = aTriangle) { sa ->
        stagedRegion(arrays, b, c, stable = sa, original = a) { sb -> body(sa, sb) }
    }
}

/**
 * Stage triangular coefficients before scaling right-hand sides by alpha:
 * aliased destination writes must not change coefficients still needed for substitution.
 */
internal inline fun prepareTriangularMatrix(
    a: DenseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    what: String,
    arrays: Scratch?,
    body: (DenseMatrix) -> Unit,
) {
    requireTriangularMatrixOperands(a, b, right, what)
    if (alpha == 0.0) {
        scaleRegion(b, 0.0)
        return
    }
    val sides = if (right) b.rows else b.cols
    if (sides == 0 || a.rows == 0) return
    stagedRegion(arrays, a, b, lower = lower, unitDiag = unitDiag) { sa -> body(sa) }
}

/** Whole and rectangular GER share panel selection and staging without allocating unaliased views. */
internal fun KoblasEngine.updateRank(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrix,
    rowOffset: Int,
    columnOffset: Int,
    rows: Int,
    columns: Int,
    scratch: Scratch?,
) {
    requireVectorWindow(x.size, xOffset, rows, xStride, "x")
    requireVectorWindow(y.size, yOffset, columns, yStride, "y")
    if (alpha == 0.0 || rows == 0 || columns == 0) return
    val destinationOffset = a.offset + rowOffset + columnOffset * a.leadingDimension
    if (x !== a.values && y !== a.values) {
        scheduleGer(alpha, x, xOffset, xStride, y, yOffset, yStride, a, destinationOffset, rows, columns)
        return
    }
    val window = if (rowOffset == 0 && columnOffset == 0 && rows == a.rows && columns == a.cols) {
        a
    } else {
        a.view(rowOffset, columnOffset, rows, columns)
    }
    val xView = DenseVector.wrap(x, xOffset, rows, xStride)
    val yView = DenseVector.wrap(y, yOffset, columns, yStride)
    stagedWindows(scratch, xView, yView, window) { sx, sy ->
        scheduleGer(
            alpha, sx.values, sx.offset, sx.stride, sy.values, sy.offset, sy.stride,
            window, window.offset, rows, columns,
        )
    }
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
            v.fill(0.0, off, off + len)
        } else {
            for (i in 0 until len) v[off + i * stride] = 0.0
        }

        beta != 1.0 -> k.kernels.scale(v, off, beta, len, stride)
    }
}
