package com.eignex.koblas.dense

/**
 * Solves `X * T = B` in place for one packed right-hand-side tile.
 *
 * [packedTriangle] is the effective, non-transposed triangle in packed-right layout: row `i` starts at
 * `triangleOff + i * tileColumns`. [x] is both B and X in packed-left layout: column `j` starts at
 * `xOff + j * tileRows`. Only [validRows] rows and [order] columns are part of the solve.
 *
 * A zero off-diagonal coefficient is skipped. Besides saving work, that preserves the triangular BLAS
 * convention already used by the portable right-side solve: a structural zero must not form `0 * Infinity`.
 * Division is performed at the pivot rather than through a stored reciprocal, because those differ when
 * a finite quotient is representable but the reciprocal overflows.
 */
@Suppress("LongParameterList") // two packed tiles, their physical shapes and the triangular flags
internal fun portableTrsmTile(
    tileRows: Int,
    tileColumns: Int,
    validRows: Int,
    order: Int,
    packedTriangle: DoubleArray,
    triangleOff: Int,
    lower: Boolean,
    unitDiag: Boolean,
    x: DoubleArray,
    xOff: Int,
) {
    if (validRows == 0 || order == 0) return
    if (lower) {
        for (j in order - 1 downTo 0) {
            dividePackedColumn(tileRows, validRows, packedTriangle, triangleOff, tileColumns, j, unitDiag, x, xOff)
            for (column in 0 until j) {
                val coefficient = packedTriangle[triangleOff + j * tileColumns + column]
                if (coefficient != 0.0) {
                    subtractPackedColumn(tileRows, validRows, x, xOff, column, j, coefficient)
                }
            }
        }
    } else {
        for (j in 0 until order) {
            dividePackedColumn(tileRows, validRows, packedTriangle, triangleOff, tileColumns, j, unitDiag, x, xOff)
            for (column in j + 1 until order) {
                val coefficient = packedTriangle[triangleOff + j * tileColumns + column]
                if (coefficient != 0.0) {
                    subtractPackedColumn(tileRows, validRows, x, xOff, column, j, coefficient)
                }
            }
        }
    }
}

/** Reference composition for a packed update followed by [portableTrsmTile], restricted to its logical edge. */
@Suppress("LongParameterList") // three packed operands, their windows, the logical edge and triangular flags
internal fun portableGemmTrsmTile(
    tileRows: Int,
    tileColumns: Int,
    depth: Int,
    validRows: Int,
    order: Int,
    packedA: DoubleArray,
    aOff: Int,
    packedB: DoubleArray,
    bOff: Int,
    packedTriangle: DoubleArray,
    triangleOff: Int,
    lower: Boolean,
    unitDiag: Boolean,
    x: DoubleArray,
    xOff: Int,
) {
    if (validRows == 0 || order == 0) return
    for (step in 0 until depth) {
        for (column in 0 until order) {
            val target = xOff + column * tileRows
            val coefficient = packedB[bOff + step * tileColumns + column]
            for (row in 0 until validRows) {
                x[target + row] -=
                    packedA[aOff + step * tileRows + row] * coefficient
            }
        }
    }
    portableTrsmTile(
        tileRows, tileColumns, validRows, order,
        packedTriangle, triangleOff, lower, unitDiag, x, xOff,
    )
}

@Suppress("LongParameterList") // physical tile shapes plus the source and destination windows
private fun dividePackedColumn(
    tileRows: Int,
    validRows: Int,
    packedTriangle: DoubleArray,
    triangleOff: Int,
    tileColumns: Int,
    column: Int,
    unitDiag: Boolean,
    x: DoubleArray,
    xOff: Int,
) {
    if (unitDiag) return
    val diagonal = packedTriangle[triangleOff + column * tileColumns + column]
    val target = xOff + column * tileRows
    for (row in 0 until validRows) x[target + row] /= diagonal
}

@Suppress("LongParameterList") // physical tile shape plus both column windows and their coefficient
private fun subtractPackedColumn(
    tileRows: Int,
    validRows: Int,
    x: DoubleArray,
    xOff: Int,
    targetColumn: Int,
    solvedColumn: Int,
    coefficient: Double,
) {
    val target = xOff + targetColumn * tileRows
    val solved = xOff + solvedColumn * tileRows
    for (row in 0 until validRows) x[target + row] -= x[solved + row] * coefficient
}
