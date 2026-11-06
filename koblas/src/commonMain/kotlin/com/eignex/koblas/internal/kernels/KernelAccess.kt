@file:Suppress("LongParameterList", "TooManyFunctions", "MaxLineLength") // Internal adapters forward primitive windows without allocating views.

package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*

/** Every built-in engine supplies the primitive kernel seam; public custom engines may implement typed operations. */
internal val KoblasEngine.kernels: EngineKernels get() = this as? EngineKernels ?: unsupported()

internal fun KoblasEngine.dot(
    a: DoubleArray,
    aOff: Int,
    b: DoubleArray,
    bOff: Int,
    len: Int,
    aStride: Int = 1,
    bStride: Int = 1,
): Double = kernels.dot(a, aOff, b, bOff, len, aStride, bStride)

internal fun KoblasEngine.axpy(
    y: DoubleArray,
    yOff: Int,
    alpha: Double,
    x: DoubleArray,
    xOff: Int,
    len: Int,
    yStride: Int = 1,
    xStride: Int = 1,
): Unit = kernels.axpy(y, yOff, alpha, x, xOff, len, yStride, xStride)

internal fun KoblasEngine.scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int = 1): Unit =
    kernels.scale(v, vOff, alpha, len, vStride)

internal fun KoblasEngine.nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double =
    kernels.nrm2(v, vOff, len, vStride)

internal fun KoblasEngine.asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double =
    kernels.asum(v, vOff, len, vStride)

internal fun KoblasEngine.iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Int =
    kernels.iamax(v, vOff, len, vStride)

internal fun KoblasEngine.rot(
    x: DoubleArray,
    xOff: Int,
    y: DoubleArray,
    yOff: Int,
    len: Int,
    c: Double,
    s: Double,
    xStride: Int = 1,
    yStride: Int = 1,
): Unit = kernels.rot(x, xOff, y, yOff, len, c, s, xStride, yStride)

internal fun KoblasEngine.swap(
    a: DoubleArray,
    aOff: Int,
    b: DoubleArray,
    bOff: Int,
    len: Int,
    aStride: Int = 1,
    bStride: Int = 1,
): Unit = kernels.swap(a, aOff, b, bOff, len, aStride, bStride)

internal fun KoblasEngine.copy(
    a: DoubleArray,
    aOff: Int,
    b: DoubleArray,
    bOff: Int,
    len: Int,
    aStride: Int = 1,
    bStride: Int = 1,
): Unit = kernels.copy(a, aOff, b, bOff, len, aStride, bStride)

internal fun KoblasEngine.sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double =
    kernels.sum(v, vOff, len, vStride)

internal fun KoblasEngine.dot(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    dense: DoubleArray,
): Double = kernels.dot(indices, indexOffset, values, valueOffset, count, dense)

internal fun KoblasEngine.dot(
    xIndices: IntArray,
    xIndexOffset: Int,
    xValues: DoubleArray,
    xValueOffset: Int,
    xCount: Int,
    yIndices: IntArray,
    yIndexOffset: Int,
    yValues: DoubleArray,
    yValueOffset: Int,
    yCount: Int,
): Double =
    kernels.dot(xIndices, xIndexOffset, xValues, xValueOffset, xCount, yIndices, yIndexOffset, yValues, yValueOffset, yCount)

internal fun KoblasEngine.axpy(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    alpha: Double,
    destination: DoubleArray,
): Unit = kernels.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)

internal fun KoblasEngine.scatter(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    destination: DoubleArray,
): Unit = kernels.scatter(indices, indexOffset, values, valueOffset, count, destination)

internal fun KoblasEngine.gather(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    source: DoubleArray,
): Unit = kernels.gather(indices, indexOffset, values, valueOffset, count, source)

internal fun KoblasEngine.gatherZero(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    source: DoubleArray,
): Unit = kernels.gatherZero(indices, indexOffset, values, valueOffset, count, source)

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
): Unit = kernels.gemv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, scratch)

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
): Unit = kernels.symv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch)

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
): Unit = kernels.ger(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch)

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
): Unit = kernels.ger(alpha, x, y, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset, scratch)

internal fun KoblasEngine.syr(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    a: DenseMatrix,
    lower: Boolean = true,
    scratch: Scratch? = null,
): Unit = kernels.syr(alpha, x, xOffset, xStride, a, lower, scratch)

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
): Unit = kernels.syr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, lower, scratch)

internal fun KoblasEngine.trsv(
    a: DenseMatrix,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
): Unit = kernels.trsv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)

internal fun KoblasEngine.trmv(
    a: DenseMatrix,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
): Unit = kernels.trmv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)

internal fun KoblasEngine.dotColumn(matrix: SparseMatrix, column: Int, dense: DoubleArray): Double =
    kernels.dotColumn(matrix, column, dense)

internal fun KoblasEngine.gemv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    transpose: Boolean = false,
    scratch: Scratch? = null,
): Unit = kernels.gemv(alpha, a, x, beta, y, transpose, scratch)

internal fun KoblasEngine.symv(
    alpha: Double,
    a: SparseMatrix,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean = true,
    scratch: Scratch? = null,
): Unit = kernels.symv(alpha, a, x, beta, y, lower, scratch)

internal fun KoblasEngine.trsv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
): Unit = kernels.trsv(a, x, lower, transpose, unitDiag, scratch)

internal fun KoblasEngine.trmv(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
): Unit = kernels.trmv(a, x, lower, transpose, unitDiag, scratch)
