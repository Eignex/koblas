@file:Suppress("LongParameterList", "TooManyFunctions") // adapters for independently addressed BLAS windows

package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*
import com.eignex.koblas.internal.operations.*

// Primitive BLAS windows adapt to the typed operation contract; arithmetic stays in the selected kernels.

internal fun KoblasEngine.gemv(
    alpha: Double,
    a: DenseMatrix,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    transpose: Boolean = false,
    scratch: Scratch? = null,
) {
    val rows = if (transpose) a.cols else a.rows
    val depth = if (transpose) a.rows else a.cols
    gemv(
        alpha,
        a,
        DenseVector.wrap(x, xOffset, depth, xStride),
        beta,
        DenseVector.wrap(y, yOffset, rows, yStride),
        transpose,
        scratch,
    )
}

internal fun KoblasEngine.symv(
    alpha: Double,
    a: DenseMatrix,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    lower: Boolean = true,
    scratch: Scratch? = null,
) {
    symv(
        alpha,
        a,
        DenseVector.wrap(x, xOffset, a.rows, xStride),
        beta,
        DenseVector.wrap(y, yOffset, a.rows, yStride),
        lower,
        scratch,
    )
}

internal fun KoblasEngine.ger(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrix,
    scratch: Scratch? = null,
) {
    ger(alpha, DenseVector.wrap(x, xOffset, a.rows, xStride), DenseVector.wrap(y, yOffset, a.cols, yStride), a, scratch)
}

internal fun KoblasEngine.ger(
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    a: DenseMatrix,
    rowOffset: Int,
    columnOffset: Int,
    rows: Int,
    columns: Int,
    xOffset: Int = 0,
    yOffset: Int = 0,
    scratch: Scratch? = null,
) {
    requireGerWindowOperands(x.size, y.size, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset)
    updateRank(alpha, x, xOffset, 1, y, yOffset, 1, a, rowOffset, columnOffset, rows, columns, scratch)
}

internal fun KoblasEngine.syr(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    a: DenseMatrix,
    lower: Boolean = true,
    scratch: Scratch? = null,
) {
    syr(alpha, DenseVector.wrap(x, xOffset, a.rows, xStride), a, lower, scratch)
}

internal fun KoblasEngine.syr2(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrix,
    lower: Boolean = true,
    scratch: Scratch? = null,
) {
    syr2(
        alpha,
        DenseVector.wrap(x, xOffset, a.rows, xStride),
        DenseVector.wrap(y, yOffset, a.rows, yStride),
        a,
        lower,
        scratch,
    )
}

internal fun KoblasEngine.trsv(
    a: DenseMatrix,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    trsv(a, DenseVector.wrap(x, xOffset, a.rows, xStride), lower, transpose, unitDiag, scratch)
}

internal fun KoblasEngine.trmv(
    a: DenseMatrix,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    trmv(a, DenseVector.wrap(x, xOffset, a.rows, xStride), lower, transpose, unitDiag, scratch)
}

internal fun KoblasEngine.dotColumn(matrix: SparseMatrix, column: Int, dense: DoubleArray): Double =
    dotColumn(matrix, column, DenseVector.wrap(dense))

internal fun KoblasEngine.gemv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean = false,
    scratch: Scratch? = null,
) {
    gemv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), transpose, scratch)
}

internal fun KoblasEngine.symv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean = true,
    scratch: Scratch? = null,
) {
    symv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), lower, scratch)
}

internal fun KoblasEngine.trsv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    trsv(a, DenseVector.wrap(x), lower, transpose, unitDiag, scratch)
}

internal fun KoblasEngine.trmv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    trmv(a, DenseVector.wrap(x), lower, transpose, unitDiag, scratch)
}
