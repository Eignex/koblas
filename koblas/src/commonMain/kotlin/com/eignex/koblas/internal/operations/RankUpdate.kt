@file:Suppress("LongParameterList") // Independent source windows and a destination rectangle.

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.scheduling.dense.scheduleGer
import com.eignex.koblas.internal.storage.requireVectorWindow

/** Whole and rectangular GER share panel selection and staging without allocating unaliased views. */
internal fun KoblasEngine.updateRank(
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    a: DenseMatrixRegion,
    rowOffset: Int,
    columnOffset: Int,
    rows: Int,
    columns: Int,
    scratch: Scratch?,
) {
    requireVectorWindow(x.size, xOffset, rows, xStride, "x")
    requireVectorWindow(y.size, yOffset, columns, yStride, "y")
    if (alpha == 0.0 || rows == 0 || columns == 0) return
    val destinationOffset = a.offset + rowOffset + columnOffset * a.leadingDimension
    if (x !== a.values && y !== a.values) {
        scheduleGer(alpha, x, xOffset, xStride, y, yOffset, yStride, a, destinationOffset, rows, columns)
        return
    }
    val window = if (rowOffset == 0 && columnOffset == 0 && rows == a.rows && columns == a.cols) {
        a
    } else {
        a.view(rowOffset, columnOffset, rows, columns)
    }
    stagedWindows(scratch, x, xOffset, rows, xStride, y, yOffset, columns, yStride, window) {
            sx,
            sxo,
            sxs,
            sy,
            syo,
            sys,
        ->
        scheduleGer(alpha, sx, sxo, sxs, sy, syo, sys, window, window.offset, rows, columns)
    }
}
