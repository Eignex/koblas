@file:Suppress("LongParameterList")

package com.eignex.koblas.dense

import kotlin.math.min

/** Applies the column runs of `A += alpha * x * y transpose`, skipping raw zero [y] coefficients. */
internal fun gerUpdate(
    kernels: DensePanelKernels,
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    a: DoubleArray,
    rows: Int,
    columns: Int,
) {
    repeat(columns) { column ->
        val coefficient = y[column]
        if (coefficient != 0.0) {
            axpyArithmetic(kernels, a, column * rows, alpha * coefficient, x, 0, rows)
        }
    }
}

/** Cache traversal shared by ordered rank-k and rank-2k fallback updates. */
internal fun blockedSymmetricRankUpdate(
    kernels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    b: DoubleArray?,
    c: DoubleArray,
    n: Int,
    depth: Int,
    lower: Boolean,
    guardZeroColumns: Boolean,
) {
    var column = 0
    while (column < n) {
        val columnEnd = min(column + LEVEL3_BLOCK_COLUMNS, n)
        var inner = 0
        while (inner < depth) {
            val innerEnd = min(inner + LEVEL3_BLOCK_DEPTH, depth)
            var p = inner
            while (p < innerEnd) {
                val sourceColumn = p * n
                var j = column
                while (j < columnEnd) {
                    val firstValue = b?.get(j + sourceColumn) ?: a[j + sourceColumn]
                    val secondValue = if (b == null) 0.0 else a[j + sourceColumn]
                    val skip = guardZeroColumns && if (b == null) {
                        firstValue == 0.0
                    } else {
                        firstValue == 0.0 && secondValue == 0.0
                    }
                    if (!skip) {
                        val firstMultiplier = alpha * firstValue
                        val secondMultiplier = alpha * secondValue
                        val triangleFrom = if (lower) j else 0
                        val triangleUntil = if (lower) n else j + 1
                        var row = triangleFrom
                        while (row < triangleUntil) {
                            val length = min(row + LEVEL3_BLOCK_ROWS, triangleUntil) - row
                            axpyArithmetic(kernels, c, row + j * n, firstMultiplier, a, row + sourceColumn, length)
                            if (b != null) {
                                axpyArithmetic(
                                    kernels,
                                    c,
                                    row + j * n,
                                    secondMultiplier,
                                    b,
                                    row + sourceColumn,
                                    length,
                                )
                            }
                            row += length
                        }
                    }
                    j++
                }
                p++
            }
            inner = innerEnd
        }
        column = columnEnd
    }
}

/** Applies the selected column runs of `A += alpha * x * x transpose`. */
internal fun syrUpdate(
    kernels: DensePanelKernels,
    alpha: Double,
    x: DoubleArray,
    a: DoubleArray,
    n: Int,
    lower: Boolean,
) {
    repeat(n) { column ->
        if (x[column] != 0.0) {
            val from = if (lower) column else 0
            val length = if (lower) n - column else column + 1
            axpyArithmetic(kernels, a, from + column * n, alpha * x[column], x, from, length)
        }
    }
}

/** Applies the selected column runs of `A += alpha * (x * y transpose + y * x transpose)`. */
internal fun syr2Update(
    kernels: DensePanelKernels,
    alpha: Double,
    x: DoubleArray,
    y: DoubleArray,
    a: DoubleArray,
    n: Int,
    lower: Boolean,
) {
    repeat(n) { column ->
        if (x[column] != 0.0 || y[column] != 0.0) {
            val from = if (lower) column else 0
            val length = if (lower) n - column else column + 1
            val matrixOffset = from + column * n
            axpyArithmetic(kernels, a, matrixOffset, alpha * y[column], x, from, length)
            axpyArithmetic(kernels, a, matrixOffset, alpha * x[column], y, from, length)
        }
    }
}
