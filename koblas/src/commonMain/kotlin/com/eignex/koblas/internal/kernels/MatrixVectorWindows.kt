@file:Suppress("LongParameterList") // a rectangular BLAS window and its operand offsets

package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*
import com.eignex.koblas.internal.operations.*

// Rectangular GER keeps a direct panel schedule without allocating unaliased views.

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
