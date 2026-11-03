@file:Suppress("LongParameterList") // Validated BLAS operands and portable matrix schedules.

package com.eignex.koblas.internal.portable.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.scaleRegion

/**
 * The product itself, over operands already staged against an overlap with the destination.
 *
 * Which path runs is the extents' answer rather than the engine's. A destination with one column or one
 * row is a matrix-vector product, so it runs as [gemv] arithmetic over the operand that is a matrix: the
 * product schedule would otherwise make a separate panel call for every destination column, which on a
 * single row is one call per entry. A single entry is a dot product either way and stays on the schedule,
 * which measured slightly ahead of the matrix-vector form for a long shared dimension.
 */
internal fun KoblasEngine.portableGemm(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    depth: Int,
    arrays: ArraySupplier?,
) {
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta)
        return
    }
    if (c.cols == 1 && c.rows > 1) {
        // The column of op(B) is a stored column, or a stored row when B is transposed.
        portableGemv(
            alpha, a, b.values, b.offset, if (transposeB) b.leadingDimension else 1,
            beta, c.values, c.offset, 1, transposeA,
        )
        return
    }
    if (c.rows == 1 && c.cols > 1) {
        // The destination row is op(B)ᵀ times the row of op(A), a stored column when A is transposed.
        portableGemv(
            alpha, b, a.values, a.offset, if (transposeA) 1 else a.leadingDimension,
            beta, c.values, c.offset, c.leadingDimension, !transposeB,
        )
        return
    }
    productWindow(
        this, alpha, a.values, a.offset, a.leadingDimension, transposeA, b.values, b.offset,
        b.leadingDimension, transposeB, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
        OutputTriangle.Full, arrays,
    )
}

internal fun KoblasEngine.portableGemmt(
    alpha: Double,
    a: DenseMatrixRegion,
    transposeA: Boolean,
    b: DenseMatrixRegion,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
) {
    val depth = if (transposeA) a.rows else a.cols
    if (c.rows == 0) return
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    productWindow(
        this, alpha, a.values, a.offset, a.leadingDimension, transposeA, b.values, b.offset,
        b.leadingDimension, transposeB, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
        selectedTriangle(
            lower,
        ),
        arrays,
    )
}

internal fun KoblasEngine.portableSymm(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    right: Boolean,
    arrays: ArraySupplier?,
) {
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    scaleRegion(c, beta, vectors = this)
    symmetricProduct(
        this, alpha, a.values, a.rows, lower, b.values, b.leadingDimension, c.values, c.leadingDimension,
        c.rows, c.cols, right, arrays, a.offset, a.leadingDimension, b.offset, c.offset,
    )
}

internal fun KoblasEngine.portableSyrk(
    alpha: Double,
    a: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
) {
    val depth = if (transpose) a.rows else a.cols
    if (c.rows == 0) return
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    productWindow(
        this, alpha, a.values, a.offset, a.leadingDimension, transpose, a.values, a.offset,
        a.leadingDimension, !transpose, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
        selectedTriangle(
            lower,
        ),
        arrays,
    )
}

internal fun KoblasEngine.portableSyr2k(
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean,
    arrays: ArraySupplier?,
) {
    val depth = if (transpose) a.rows else a.cols
    if (c.rows == 0) return
    if (alpha == 0.0 || depth == 0) {
        scaleRegion(c, beta, lower)
        return
    }
    val selected = selectedTriangle(lower)
    productWindow(
        this, alpha, a.values, a.offset, a.leadingDimension, transpose, b.values, b.offset,
        b.leadingDimension, !transpose, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
        selected, arrays,
    )
    productWindow(
        this, alpha, b.values, b.offset, b.leadingDimension, transpose, a.values, a.offset,
        a.leadingDimension, !transpose, 1.0, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
        selected, arrays,
    )
}

internal fun KoblasEngine.portableTriangularMatrix(
    alpha: Double,
    a: DenseMatrixRegion,
    structure: MatrixStructure,
    transpose: Boolean,
    b: DenseMatrixRegion,
    right: Boolean,
    solve: Boolean,
    arrays: ArraySupplier?,
) {
    if (alpha == 0.0) {
        scaleRegion(b, 0.0)
        return
    }
    triangularMatrix(
        this, a.values, a.rows, b.values, b.leadingDimension, if (right) b.rows else b.cols,
        structure.isLower, transpose, structure.isUnitDiagonal, right, alpha, solve, arrays, a.offset,
        a.leadingDimension, b.offset,
    )
}
