@file:Suppress("LongParameterList")

package com.eignex.koblas.dense

/**
 * Adds `alpha * op(A) * x` to [y] for a contiguous column-major matrix. The four-column path keeps its
 * reductions and coefficient/writeback work together; scalar tails retain the same arithmetic order.
 */
internal fun denseGemvUpdate(
    vectorKernels: DenseVectorKernels,
    panelKernels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    rows: Int,
    columns: Int,
    x: DoubleArray,
    y: DoubleArray,
    transpose: Boolean,
) {
    var column = 0
    val groupedEnd = columns - 3
    if (!transpose) {
        while (column < groupedEnd) {
            panelKernels.axpy4(
                y, 0, a, column * rows, rows,
                alpha * x[column], alpha * x[column + 1], alpha * x[column + 2], alpha * x[column + 3], rows,
            )
            column += 4
        }
        while (column < columns) {
            axpyArithmetic(panelKernels, y, 0, alpha * x[column], a, column * rows, rows)
            column++
        }
        return
    }
    while (column < groupedEnd) {
        val y0 = y[column]
        val y1 = y[column + 1]
        val y2 = y[column + 2]
        val y3 = y[column + 3]
        panelKernels.dot4(a, column * rows, rows, x, 0, rows, y, column)
        y[column] = y0 + alpha * y[column]
        y[column + 1] = y1 + alpha * y[column + 1]
        y[column + 2] = y2 + alpha * y[column + 2]
        y[column + 3] = y3 + alpha * y[column + 3]
        column += 4
    }
    while (column < columns) {
        y[column] += alpha * vectorKernels.dot(a, column * rows, x, 0, rows)
        column++
    }
}

/** Reduces four matrix columns against one vector run and adds their scaled results to four destinations. */
internal fun dot4Writeback(
    panelKernels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    aOff: Int,
    stride: Int,
    b: DoubleArray,
    bOff: Int,
    length: Int,
    c: DoubleArray,
    cOff: Int,
    sums: DoubleArray,
) {
    panelKernels.dot4(a, aOff, stride, b, bOff, length, sums, 0)
    c[cOff] += alpha * sums[0]
    c[cOff + 1] += alpha * sums[1]
    c[cOff + 2] += alpha * sums[2]
    c[cOff + 3] += alpha * sums[3]
}

/** Reduces one pair of runs and adds its scaled dot to one destination entry. */
internal fun dotWriteback(
    vectorKernels: DenseVectorKernels,
    alpha: Double,
    a: DoubleArray,
    aOff: Int,
    b: DoubleArray,
    bOff: Int,
    length: Int,
    c: DoubleArray,
    cOff: Int,
) {
    c[cOff] += alpha * vectorKernels.dot(a, aOff, b, bOff, length)
}
