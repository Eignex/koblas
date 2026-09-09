package com.eignex.koblas.dense

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
