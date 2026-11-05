@file:Suppress("LongParameterList") // a routine's operands, its multipliers and the scratch it may borrow

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.requireVectorWindow
import com.eignex.koblas.internal.storage.windowsOverlap

/*
 * What every implementation of a dense routine does around its arithmetic after checking the routine's shapes:
 * settle the calls with nothing to compute or a zero multiplier, and snapshot an input that shares the
 * destination's storage. The shared schedule and a vendor binding run their arithmetic inside the same
 * preparation, so a call reaches either one with the same operands. Vector operands are array windows, so a
 * call whose operands share no storage allocates nothing here.
 */

/**
 * `y = alpha · op(A) · x + beta · y`, for `gemv` and, from one [lower] triangle, `symv`. [body] receives the
 * stable matrix and the stable `x` window.
 */
internal inline fun prepareMatrixVector(
    vectors: KoblasEngine,
    alpha: Double,
    a: DenseMatrixRegion,
    transpose: Boolean,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    lower: Boolean?,
    arrays: Scratch?,
    body: (DenseMatrixRegion, DoubleArray, Int, Int) -> Unit,
) {
    val rows = if (transpose) a.cols else a.rows
    val depth = if (transpose) a.rows else a.cols
    requireVectorWindow(x.size, xOffset, depth, xStride, "x")
    requireVectorWindow(y.size, yOffset, rows, yStride, "y")
    if (rows == 0) return
    if (alpha == 0.0 || depth == 0) {
        applyBeta(vectors, y, yOffset, rows, beta, yStride)
        return
    }
    if (a.values !== y && x !== y) return body(a, x, xOffset, xStride)
    stagedRegionAgainst(arrays, a, y, yOffset, rows, yStride, lower = lower) { sa ->
        val aliased = windowsOverlap(x, xOffset, depth, xStride, y, yOffset, rows, yStride)
        stagedWindow(arrays, x, xOffset, depth, xStride, aliased) { sx, so, ss -> body(sa, sx, so, ss) }
    }
}

/**
 * `A = A + alpha · x · yᵀ` and its symmetric forms, with `x` spanning `A`'s rows and `y` its columns. [body]
 * receives the stable windows; identical windows share one snapshot.
 */
internal inline fun prepareRankUpdate(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrixRegion,
    arrays: Scratch?,
    body: (DoubleArray, Int, Int, DoubleArray, Int, Int) -> Unit,
) {
    requireVectorWindow(x.size, xOffset, a.rows, xStride, "x")
    requireVectorWindow(y.size, yOffset, a.cols, yStride, "y")
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
    if (x !== a.values && y !== a.values) return body(x, xOffset, xStride, y, yOffset, yStride)
    stagedWindows(arrays, x, xOffset, a.rows, xStride, y, yOffset, a.cols, yStride, a, body)
}

/**
 * `op(T) · x = b` or `x = op(T) · x` in place. The substitution overwrites `x` as it goes, so a triangle
 * sharing that storage is snapshotted first.
 */
internal inline fun prepareTriangularVector(
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    unitDiag: Boolean,
    what: String,
    arrays: Scratch?,
    body: (DenseMatrixRegion) -> Unit,
) {
    requireSquare(a, what)
    val n = a.rows
    requireVectorWindow(x.size, xOffset, n, xStride, "$what x")
    if (n == 0) return
    if (a.values !== x) return body(a)
    stagedRegionAgainst(arrays, a, x, xOffset, n, xStride, lower = lower, unitDiag = unitDiag) { body(it) }
}

/**
 * A product into [c] over [depth] shared entries. A zero [alpha] or depth scales [c], or its [selected]
 * triangle, without reading an operand. [aTriangle] limits [a]'s snapshot to the triangle the routine reads.
 */
internal inline fun prepareProduct(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    beta: Double,
    c: DenseMatrixRegion,
    depth: Int,
    selected: Boolean?,
    aTriangle: Boolean?,
    vectors: KoblasEngine?,
    arrays: Scratch?,
    body: (DenseMatrixRegion, DenseMatrixRegion) -> Unit,
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
 * `B = alpha · op(T)⁻¹ · B` or `B = alpha · op(T) · B`, from either side.
 *
 * The staging comes before any scaling. `alpha` is spent on the right-hand sides before the substitution,
 * and where the triangle shares their storage that write would otherwise reach the coefficients the
 * substitution is about to read.
 */
internal inline fun prepareTriangularMatrix(
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    lower: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    what: String,
    arrays: Scratch?,
    body: (DenseMatrixRegion) -> Unit,
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
