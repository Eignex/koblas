package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import kotlin.math.min
import kotlin.math.nextDown

/** Whether a packed traversal preserves this call's zero, non-finite and overflow behavior. */
internal fun packedTrmmSupports(a: DenseMatrix, b: DenseMatrix, lower: Boolean, unitDiagonal: Boolean): Boolean {
    val maximumSource = finiteMaxAbs(b.data) ?: return false
    val order = a.rows
    val maximumTriangle = triangularMultiplyMaxAbs(a.data, order, lower, unitDiagonal) ?: return false
    if (order == 0 || maximumSource == 0.0 || maximumTriangle == 0.0) return true

    // Every partial sum in either traversal is bounded by the sum of absolute products. Keeping that sum
    // finite prevents packing and fused multiply-adds from hiding or introducing intermediate overflow.
    val productLimit = (Double.MAX_VALUE / order).nextDown()
    return maximumTriangle <= (productLimit / maximumSource).nextDown()
}

/** Packs both immutable inputs, then evaluates only each triangular output tile's live depth range. */
@Suppress("LongParameterList") // BLAS flags plus the two operands and scratch
internal fun packedTrmmCore(
    kernels: PackedKernels,
    a: DenseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiagonal: Boolean,
    right: Boolean,
    workspace: Workspace?,
) {
    val order = a.rows
    val tileRows = kernels.gemmTileRows
    val tileColumns = kernels.gemmTileCols
    val packedTriangleSize = if (right) {
        packedRightSize(order, order, tileColumns)
    } else {
        packedLeftSize(order, order, tileRows)
    }
    val packedSourceSize = if (right) {
        packedLeftSize(b.rows, order, tileRows)
    } else {
        packedRightSize(order, b.cols, tileColumns)
    }
    workspace.borrow(packedTriangleSize) { packedTriangle ->
        workspace.borrow(packedSourceSize) { packedSource ->
            workspace.borrow(tileRows * tileColumns) { tile ->
                if (right) {
                    packRightLayout(
                        a.data, a.rows, packedTriangle, 0, order, order, 0, 0, transpose,
                        PackedPanelStructure.Triangular, lower, unitDiagonal, tileColumns,
                    )
                    packLeftLayout(
                        b.data, b.rows, packedSource, 0, b.rows, order, 0, 0, false, 1.0,
                        PackedPanelStructure.General, false, false, tileRows,
                    )
                    b.data.fill(0.0)
                    packedRightTrmm(
                        kernels,
                        packedSource,
                        packedTriangle,
                        b,
                        order,
                        if (transpose) !lower else lower,
                        tile,
                    )
                } else {
                    packLeftLayout(
                        a.data, a.rows, packedTriangle, 0, order, order, 0, 0, transpose, 1.0,
                        PackedPanelStructure.Triangular, lower, unitDiagonal, tileRows,
                    )
                    packRightLayout(
                        b.data, b.rows, packedSource, 0, order, b.cols, 0, 0, false,
                        PackedPanelStructure.General, false, false, tileColumns,
                    )
                    b.data.fill(0.0)
                    packedLeftTrmm(
                        kernels,
                        packedTriangle,
                        packedSource,
                        b,
                        order,
                        if (transpose) !lower else lower,
                        tile,
                    )
                }
            }
        }
    }
}

private fun packedRightTrmm(
    kernels: PackedKernels,
    packedSource: DoubleArray,
    packedTriangle: DoubleArray,
    destination: DenseMatrix,
    order: Int,
    effectiveLower: Boolean,
    tile: DoubleArray,
) {
    val tileRows = kernels.gemmTileRows
    val tileColumns = kernels.gemmTileCols
    var columnStart = 0
    while (columnStart < order) {
        val validColumns = min(tileColumns, order - columnStart)
        val depthStart = if (effectiveLower) columnStart else 0
        val depth = if (effectiveLower) order - columnStart else columnStart + validColumns
        val trianglePanel = columnStart / tileColumns * order * tileColumns
        var rowStart = 0
        while (rowStart < destination.rows) {
            val validRows = min(tileRows, destination.rows - rowStart)
            val sourcePanel = rowStart / tileRows * order * tileRows
            writePackedProductTile(
                kernels, depth, packedSource, sourcePanel + depthStart * tileRows,
                packedTriangle, trianglePanel + depthStart * tileColumns,
                destination.data, rowStart + columnStart * destination.rows, destination.rows,
                validRows, validColumns, tile,
            )
            rowStart += tileRows
        }
        columnStart += tileColumns
    }
}

private fun packedLeftTrmm(
    kernels: PackedKernels,
    packedTriangle: DoubleArray,
    packedSource: DoubleArray,
    destination: DenseMatrix,
    order: Int,
    effectiveLower: Boolean,
    tile: DoubleArray,
) {
    val tileRows = kernels.gemmTileRows
    val tileColumns = kernels.gemmTileCols
    var columnStart = 0
    while (columnStart < destination.cols) {
        val validColumns = min(tileColumns, destination.cols - columnStart)
        val sourcePanel = columnStart / tileColumns * order * tileColumns
        var rowStart = 0
        while (rowStart < order) {
            val validRows = min(tileRows, order - rowStart)
            val depthStart = if (effectiveLower) 0 else rowStart
            val depth = if (effectiveLower) rowStart + validRows else order - rowStart
            val trianglePanel = rowStart / tileRows * order * tileRows
            writePackedProductTile(
                kernels, depth, packedTriangle, trianglePanel + depthStart * tileRows,
                packedSource, sourcePanel + depthStart * tileColumns,
                destination.data, rowStart + columnStart * order, order,
                validRows, validColumns, tile,
            )
            rowStart += tileRows
        }
        columnStart += tileColumns
    }
}
