package com.eignex.koblas.internal.operands

import com.eignex.koblas.*
import com.eignex.koblas.engine.*

/** Reads a custom triangle once, excluding the unselected half and an implicit diagonal. */
internal inline fun <T> triangleOperand(
    arrays: Scratch?,
    a: Matrix,
    lower: Boolean,
    unitDiag: Boolean,
    block: (DenseMatrixRegion) -> T,
): T {
    if (a is DenseMatrixRegion) return block(a)
    return arrays.borrow(DenseMatrix.entryCount(a.rows, a.cols)) { values ->
        copyTriangle(a.rows, a.cols, lower, unitDiag, values) { i, j -> a[i, j] }
        block(DenseMatrix.wrap(a.rows, a.cols, values))
    }
}

/**
 * Writes the [lower] or upper triangle of a [rows] by [columns] operand, or all of it when [lower] is null,
 * into [copy] with [rows] as its leading dimension. A [unitDiag] diagonal is written as 1 without calling
 * [entry]; the other triangle is neither read nor written.
 */
internal inline fun copyTriangle(
    rows: Int,
    columns: Int,
    lower: Boolean?,
    unitDiag: Boolean,
    copy: DoubleArray,
    entry: (i: Int, j: Int) -> Double,
) {
    for (j in 0 until columns) {
        val first = if (lower == true) j else 0
        val last = if (lower == false && j < rows) j + 1 else rows
        for (i in first until last) copy[i + j * rows] = if (unitDiag && i == j) 1.0 else entry(i, j)
    }
}

/**
 * The triangle and diagonal a genuinely triangular operand declares.
 *
 * Only the four triangular routines use this. A symmetric destination takes [symmetricStructure] instead, even
 * though both spellings reach the same CBLAS `uplo`: a rank update's destination is stored as a symmetric
 * matrix is, and calling it triangular would declare the other half to be zeros it is not.
 */
internal fun triangle(lower: Boolean, unitDiag: Boolean): MatrixStructure = when {
    unitDiag && lower -> MatrixStructure.UnitLower
    unitDiag -> MatrixStructure.UnitUpper
    lower -> MatrixStructure.TriangularLower
    else -> MatrixStructure.TriangularUpper
}

/** The stored triangle a symmetric operand declares, shared with the callers that state it themselves. */
internal fun symmetricStructure(lower: Boolean): MatrixStructure =
    if (lower) MatrixStructure.SymmetricLower else MatrixStructure.SymmetricUpper
