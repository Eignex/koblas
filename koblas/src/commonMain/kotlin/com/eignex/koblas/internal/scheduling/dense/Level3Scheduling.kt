@file:Suppress("LongParameterList") // Validated BLAS operands and portable matrix schedules.

package com.eignex.koblas.internal.scheduling.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.operations.*

/*
 * Every entry point here runs inside its routine's preparation, which has already settled an empty
 * destination, an empty shared dimension and a zero alpha, so each one starts at the arithmetic.
 */

/**
 * The product itself, over operands already staged against an overlap with the destination.
 *
 * Which path runs is the extents' answer rather than the engine's. A destination with one column or one
 * row is a matrix-vector product, so it runs as [KoblasEngine.gemv] arithmetic over the matrix operand: the
 * product schedule would otherwise make a separate panel call for every destination column, which on a
 * single row is one call per entry. A single entry is a dot product either way and stays on the schedule,
 * which measured slightly ahead of the matrix-vector form for a long shared dimension.
 */
internal fun KoblasEngine.scheduleGemm(
    alpha: Double,
    a: DenseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
    depth: Int,
    arrays: Scratch?,
) {
    if (c.cols == 1 && c.rows > 1) {
        // The column of op(B) is a stored column, or a stored row when B is transposed.
        scheduleGemv(
            alpha, a, b.values, b.offset, if (transposeB) b.leadingDimension else 1,
            beta, c.values, c.offset, 1, transposeA,
        )
        return
    }
    if (c.rows == 1 && c.cols > 1) {
        // The destination row is op(B)ᵀ times the row of op(A), a stored column when A is transposed.
        scheduleGemv(
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

internal fun KoblasEngine.scheduleGemmt(
    alpha: Double,
    a: DenseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    arrays: Scratch?,
) {
    val depth = if (transposeA) a.rows else a.cols
    productWindow(
        this, alpha, a.values, a.offset, a.leadingDimension, transposeA, b.values, b.offset,
        b.leadingDimension, transposeB, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
        selectedTriangle(lower), arrays,
    )
}

internal fun KoblasEngine.scheduleSymm(
    alpha: Double,
    a: DenseMatrix,
    b: DenseMatrix,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    right: Boolean,
    arrays: Scratch?,
) {
    scaleRegion(c, beta, vectors = this)
    symmetricProduct(
        this, alpha, a.values, a.rows, lower, b.values, b.leadingDimension, c.values, c.leadingDimension,
        c.rows, c.cols, right, arrays, a.offset, a.leadingDimension, b.offset, c.offset,
    )
}

internal fun KoblasEngine.scheduleSyrk(
    alpha: Double,
    a: DenseMatrix,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    arrays: Scratch?,
) {
    val depth = if (transpose) a.rows else a.cols
    productWindow(
        this, alpha, a.values, a.offset, a.leadingDimension, transpose, a.values, a.offset,
        a.leadingDimension, !transpose, beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
        selectedTriangle(lower), arrays,
    )
}

internal fun KoblasEngine.scheduleSyr2k(
    alpha: Double,
    a: DenseMatrix,
    b: DenseMatrix,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean,
    arrays: Scratch?,
) {
    val depth = if (transpose) a.rows else a.cols
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

internal fun KoblasEngine.scheduleTriangularMatrix(
    alpha: Double,
    a: DenseMatrix,
    structure: MatrixStructure,
    transpose: Boolean,
    b: DenseMatrix,
    right: Boolean,
    solve: Boolean,
    arrays: Scratch?,
) {
    triangularMatrix(
        this, a.values, a.rows, b.values, b.leadingDimension, if (right) b.rows else b.cols,
        structure.isLower, transpose, structure.isUnitDiagonal, right, alpha, solve, arrays, a.offset,
        a.leadingDimension, b.offset,
    )
}
