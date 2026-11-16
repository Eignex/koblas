@file:Suppress("LongParameterList") // Validated BLAS operands and portable matrix schedules.

package com.eignex.koblas.internal.scheduling.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.KernelProvider
import com.eignex.koblas.internal.operations.*

/*
 * Every entry point here runs inside its routine's preparation, which has already settled an empty
 * destination, an empty shared dimension and a zero alpha, so each one starts at the arithmetic.
 */

/**
 * The product itself, over operands already staged against an overlap with the destination.
 *
 * Which path runs is the extents' answer rather than the engine's. A destination with one column or one
 * row is a matrix-vector product, so it runs as [scheduleGemv] arithmetic over the matrix operand: the
 * product schedule would otherwise make a separate panel call for every destination column, which on a
 * single row is one call per entry. A single entry is a dot product either way and stays on the schedule,
 * which measured slightly ahead of the matrix-vector form for a long shared dimension.
 */
internal fun KernelProvider.scheduleGemm(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    transposeA: Boolean,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    transposeB: Boolean,
    depth: Int,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    rows: Int,
    columns: Int,
    arrays: Scratch?,
) {
    if (columns == 1 && rows > 1) {
        // The column of op(B) is a stored column, or a stored row when B is transposed.
        scheduleGemv(
            alpha, a, aOffset, lda, if (transposeA) depth else rows, if (transposeA) rows else depth,
            b, bOffset, if (transposeB) ldb else 1,
            beta, c, cOffset, 1, transposeA,
        )
        return
    }
    if (rows == 1 && columns > 1) {
        // The destination row is op(B)ᵀ times the row of op(A), a stored column when A is transposed.
        scheduleGemv(
            alpha, b, bOffset, ldb, if (transposeB) columns else depth, if (transposeB) depth else columns,
            a, aOffset, if (transposeA) 1 else lda,
            beta, c, cOffset, ldc, !transposeB,
        )
        return
    }
    productWindow(
        this, alpha, a, aOffset, lda, transposeA, b, bOffset, ldb, transposeB,
        beta, c, cOffset, ldc, rows, columns, depth, OutputTriangle.Full, arrays,
    )
}

internal fun KernelProvider.scheduleGemmt(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    transposeA: Boolean,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    transposeB: Boolean,
    depth: Int,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    rows: Int,
    columns: Int,
    lower: Boolean,
    arrays: Scratch?,
) {
    productWindow(
        this, alpha, a, aOffset, lda, transposeA, b, bOffset, ldb, transposeB,
        beta, c, cOffset, ldc, rows, columns, depth, selectedTriangle(lower), arrays,
    )
}

internal fun KernelProvider.scheduleSymm(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    order: Int,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    rows: Int,
    columns: Int,
    lower: Boolean,
    right: Boolean,
    arrays: Scratch?,
) {
    scaleRegion(c, cOffset, rows, columns, ldc, beta, vectors = this)
    symmetricProduct(
        this, alpha, a, order, lower, b, ldb, c, ldc,
        rows, columns, right, arrays, aOffset, lda, bOffset, cOffset,
    )
}

internal fun KernelProvider.scheduleSyrk(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    transpose: Boolean,
    depth: Int,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    rows: Int,
    columns: Int,
    lower: Boolean,
    arrays: Scratch?,
) {
    productWindow(
        this, alpha, a, aOffset, lda, transpose, a, aOffset, lda, !transpose,
        beta, c, cOffset, ldc, rows, columns, depth, selectedTriangle(lower), arrays,
    )
}

internal fun KernelProvider.scheduleSyr2k(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    transpose: Boolean,
    depth: Int,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    rows: Int,
    columns: Int,
    lower: Boolean,
    arrays: Scratch?,
) {
    val selected = selectedTriangle(lower)
    productWindow(
        this, alpha, a, aOffset, lda, transpose, b, bOffset, ldb, !transpose,
        beta, c, cOffset, ldc, rows, columns, depth, selected, arrays,
    )
    productWindow(
        this, alpha, b, bOffset, ldb, transpose, a, aOffset, lda, !transpose,
        1.0, c, cOffset, ldc, rows, columns, depth, selected, arrays,
    )
}

internal fun KernelProvider.scheduleTriangularMatrix(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    order: Int,
    structure: MatrixStructure,
    transpose: Boolean,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    rows: Int,
    columns: Int,
    right: Boolean,
    solve: Boolean,
    arrays: Scratch?,
) {
    triangularMatrix(
        this, a, order, b, ldb, if (right) rows else columns,
        structure.isLower, transpose, structure.isUnitDiagonal, right, alpha, solve, arrays, aOffset,
        lda, bOffset,
    )
}
