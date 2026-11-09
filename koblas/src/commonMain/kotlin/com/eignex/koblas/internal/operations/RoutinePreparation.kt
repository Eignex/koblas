@file:Suppress("LongParameterList") // a routine's operands, its multipliers and the scratch it may borrow

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.storage.*

/*
 * What every implementation of a dense routine does around its arithmetic after checking the routine's shapes:
 * settle the calls with nothing to compute or a zero multiplier, and snapshot an input that shares the
 * destination's storage. The shared schedule and a vendor binding run their arithmetic inside the same
 * preparation, so a call reaches either one with the same operands. Vector views retain their addressing
 * until the arithmetic boundary; only overlapping inputs need a snapshot.
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
 * `B = alpha · op(T)⁻¹ · B` or `B = alpha · op(T) · B`, from either side.
 *
 * The staging comes before any scaling. `alpha` is spent on the right-hand sides before the substitution,
 * and where the triangle shares their storage that write would otherwise reach the coefficients the
 * substitution is about to read.
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
