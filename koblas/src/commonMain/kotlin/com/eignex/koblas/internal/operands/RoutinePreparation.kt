@file:Suppress("LongParameterList", "TooManyFunctions") // one preparation per BLAS routine, with its operands

package com.eignex.koblas.internal.operands

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.dense.applyBeta

/*
 * What every implementation of a dense routine does before its arithmetic: check the operands, settle the
 * calls with nothing to compute or a zero multiplier, and snapshot an input that shares the destination's
 * storage. The portable schedule and a vendor binding run their arithmetic inside the same preparation, so a
 * call reaches either one with the same operands. Vector operands are array windows, so a call whose
 * operands share no storage allocates nothing here.
 */

/** `y = alpha · op(A) · x + beta · y`. [body] receives the stable matrix and the stable `x` window. */
internal inline fun prepareGemv(
    vectors: KoblasEngine,
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    transpose: Boolean,
    arrays: ArraySupplier?,
    body: (DenseMatrixRegion, DoubleArray, Int, Int) -> Unit,
) {
    val rows = if (transpose) a.cols else a.rows
    val depth = if (transpose) a.rows else a.cols
    requireVectorWindow(x.size, xOffset, depth, xStride, "gemv x")
    requireVectorWindow(y.size, yOffset, rows, yStride, "gemv y")
    if (rows == 0) return
    if (alpha == 0.0 || depth == 0) {
        applyBeta(vectors, y, yOffset, rows, beta, yStride)
        return
    }
    if (a.values !== y && x !== y) return body(a, x, xOffset, xStride)
    stagedRegionAgainst(arrays, a, y, yOffset, rows, yStride) { sa ->
        val aliased = windowsOverlap(x, xOffset, depth, xStride, y, yOffset, rows, yStride)
        stagedWindow(arrays, x, xOffset, depth, xStride, aliased) { sx, so, ss -> body(sa, sx, so, ss) }
    }
}

/** `y = alpha · A · x + beta · y` for a symmetric `A` read from one triangle. */
internal inline fun prepareSymv(
    vectors: KoblasEngine,
    alpha: Double,
    a: DenseMatrixRegion,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    lower: Boolean,
    arrays: ArraySupplier?,
    body: (DenseMatrixRegion, DoubleArray, Int, Int) -> Unit,
) {
    requireSquare(a, "symv")
    val n = a.rows
    requireVectorWindow(x.size, xOffset, n, xStride, "symv x")
    requireVectorWindow(y.size, yOffset, n, yStride, "symv y")
    if (n == 0) return
    if (alpha == 0.0) {
        applyBeta(vectors, y, yOffset, n, beta, yStride)
        return
    }
    if (a.values !== y && x !== y) return body(a, x, xOffset, xStride)
    stagedRegionAgainst(arrays, a, y, yOffset, n, yStride, lower = lower) { sa ->
        val aliased = windowsOverlap(x, xOffset, n, xStride, y, yOffset, n, yStride)
        stagedWindow(arrays, x, xOffset, n, xStride, aliased) { sx, so, ss -> body(sa, sx, so, ss) }
    }
}

/** `A = A + alpha · x · yᵀ`. [body] receives the stable `x` and `y` windows. */
internal inline fun prepareGer(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrixRegion,
    arrays: ArraySupplier?,
    body: (DoubleArray, Int, Int, DoubleArray, Int, Int) -> Unit,
) {
    requireVectorWindow(x.size, xOffset, a.rows, xStride, "ger x")
    requireVectorWindow(y.size, yOffset, a.cols, yStride, "ger y")
    if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
    if (x !== a.values && y !== a.values) return body(x, xOffset, xStride, y, yOffset, yStride)
    stagedWindows(arrays, x, xOffset, a.rows, xStride, y, yOffset, a.cols, yStride, a, body)
}

/** `A = A + alpha · x · xᵀ` in one triangle. */
internal inline fun prepareSyr(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    a: DenseMatrixRegion,
    arrays: ArraySupplier?,
    body: (DoubleArray, Int, Int) -> Unit,
) {
    requireSquare(a, "syr")
    val n = a.rows
    requireVectorWindow(x.size, xOffset, n, xStride, "syr x")
    if (alpha == 0.0 || n == 0) return
    if (x !== a.values) return body(x, xOffset, xStride)
    stagedWindow(arrays, x, xOffset, n, xStride, regionWindowOverlap(a, x, xOffset, n, xStride), body)
}

/** `A = A + alpha · (x · yᵀ + y · xᵀ)` in one triangle. */
internal inline fun prepareSyr2(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrixRegion,
    arrays: ArraySupplier?,
    body: (DoubleArray, Int, Int, DoubleArray, Int, Int) -> Unit,
) {
    requireSquare(a, "syr2")
    val n = a.rows
    requireVectorWindow(x.size, xOffset, n, xStride, "syr2 x")
    requireVectorWindow(y.size, yOffset, n, yStride, "syr2 y")
    if (alpha == 0.0 || n == 0) return
    if (x !== a.values && y !== a.values) return body(x, xOffset, xStride, y, yOffset, yStride)
    stagedWindows(arrays, x, xOffset, n, xStride, y, yOffset, n, yStride, a, body)
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
    arrays: ArraySupplier?,
    body: (DenseMatrixRegion) -> Unit,
) {
    requireSquare(a, what)
    val n = a.rows
    requireVectorWindow(x.size, xOffset, n, xStride, "$what x")
    if (n == 0) return
    if (a.values !== x) return body(a)
    stagedRegionAgainst(arrays, a, x, xOffset, n, xStride, lower = lower, unitDiag = unitDiag) { body(it) }
}

/** `C = alpha · op(A) · op(B) + beta · C`. [body] receives the stable operands and the shared depth. */
internal inline fun prepareGemm(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    arrays: ArraySupplier?,
    body: (DenseMatrixRegion, DenseMatrixRegion, Int) -> Unit,
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    // Return before borrowing operand-sized scratch for an empty destination.
    if (c.rows == 0 || c.cols == 0) return
    val depth = if (transposeA) a.rows else a.cols
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta)
        return
    }
    stagedRegion(arrays, a, c) { sa ->
        stagedRegion(arrays, b, c, stable = sa, original = a) { sb -> body(sa, sb, depth) }
    }
}

/** [prepareGemm] for a product written to one triangle of a square destination. */
internal inline fun prepareGemmt(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
    body: (DenseMatrixRegion, DenseMatrixRegion) -> Unit,
) {
    requireGemmtOperands(a, transposeA, b, transposeB, c)
    // An empty destination needs no staging, even when operands share an empty array.
    if (c.rows == 0) return
    val depth = if (transposeA) a.rows else a.cols
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    stagedRegion(arrays, a, c) { sa ->
        stagedRegion(arrays, b, c, stable = sa, original = a) { sb -> body(sa, sb) }
    }
}

/** `C = alpha · A · B + beta · C`, or `alpha · B · A + beta · C` from the right, for a symmetric `A`. */
internal inline fun prepareSymm(
    vectors: KoblasEngine,
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    right: Boolean,
    arrays: ArraySupplier?,
    body: (DenseMatrixRegion, DenseMatrixRegion) -> Unit,
) {
    requireSymmOperands(a, b, c, right)
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) {
        scaleRegion(c, beta, vectors = vectors)
        return
    }
    // One operand serving both sides needs a full snapshot, since the second reads the triangle the first skips.
    stagedRegion(arrays, a, c, lower = if (sameRegion(a, b)) null else lower) { sa ->
        stagedRegion(arrays, b, c, stable = sa, original = a) { sb -> body(sa, sb) }
    }
}

/** `C = alpha · op(A) · op(A)ᵀ + beta · C` in one triangle. */
internal inline fun prepareSyrk(
    alpha: Double,
    a: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
    body: (DenseMatrixRegion) -> Unit,
) {
    requireSyrkOperands(a, transpose, c)
    if (c.rows == 0) return
    val depth = if (transpose) a.rows else a.cols
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    stagedRegion(arrays, a, c) { sa -> body(sa) }
}

/** `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` in one triangle. */
internal inline fun prepareSyr2k(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
    body: (DenseMatrixRegion, DenseMatrixRegion) -> Unit,
) {
    requireSyr2kOperands(a, b, transpose, c)
    if (c.rows == 0) return
    val depth = if (transpose) a.rows else a.cols
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    stagedRegion(arrays, a, c) { sa ->
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
    arrays: ArraySupplier?,
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
