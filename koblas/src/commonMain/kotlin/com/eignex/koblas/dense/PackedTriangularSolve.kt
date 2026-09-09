package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.nextUp

/** Whether packing can preserve this call's observable zero and non-finite arithmetic. */
internal fun packedTrsmSupports(a: DenseMatrix, b: DenseMatrix, lower: Boolean, unitDiag: Boolean): Boolean {
    var bound = 0.0
    for (value in b.data) {
        if (!value.isFinite()) return false
        bound = maxOf(bound, abs(value))
    }
    val n = a.rows
    var minimumDiagonal = if (unitDiag) 1.0 else Double.POSITIVE_INFINITY
    var maximumOffDiagonal = 0.0
    for (column in 0 until n) {
        val from = if (lower) column else 0
        val to = if (lower) n else column + 1
        for (row in from until to) {
            if (unitDiag && row == column) continue
            val value = a.data[row + column * n]
            // Left non-transposed substitution skips a zero RHS pivot, including a singular diagonal.
            // The normalized right-side leaf always divides, so singular triangles keep the reference walk.
            if (!value.isFinite() || value == 0.0) return false
            if (row == column) {
                minimumDiagonal = minOf(minimumDiagonal, abs(value))
            } else {
                maximumOffDiagonal = maxOf(maximumOffDiagonal, abs(value))
            }
        }
    }
    // Packing changes dot/subtraction order. Bound the residual and solved entries after every pivot so
    // cancellation cannot hide an overflow in either walk. Outward rounding also covers fused updates;
    // headroom keeps the bound itself away from overflow. Dividing directly admits subnormal diagonals.
    val limit = Double.MAX_VALUE / 4.0
    repeat(n) {
        val solvedBound = (bound / minimumDiagonal).nextUp()
        bound = (bound + (maximumOffDiagonal * solvedBound).nextUp()).nextUp()
        if (solvedBound > limit || bound > limit) return false
    }
    return true
}

/**
 * Normalizes every BLAS side/transpose variant to X * T = B, packs both matrices once, then walks
 * [PackedKernels.gemmTrsmTile] between packed diagonal solves.
 */
@Suppress("LongParameterList") // BLAS flags plus the two matrix operands and workspace
internal fun packedTrsmCore(
    kernels: PackedKernels,
    a: DenseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    workspace: Workspace?,
) {
    val order = a.rows
    val rows = if (right) b.rows else b.cols
    val tileRows = kernels.gemmTileRows
    val tileColumns = kernels.gemmTileCols
    val transposeTriangle = if (right) transpose else !transpose
    val effectiveLower = if (transposeTriangle) !lower else lower
    val packedXSize = packedLeftSize(rows, order, tileRows)
    val packedTriangleSize = packedRightSize(order, order, tileColumns)
    workspace.borrow(packedTriangleSize) { packedTriangle ->
        workspace.borrow(packedXSize) { packedX ->
            packRightPanel(
                a,
                packedTriangle,
                order,
                order,
                0,
                0,
                transposeTriangle,
                0,
                null,
                PackedPanelStructure.Triangular,
                lower,
                unitDiag,
                tileColumns,
            )
            packLeftPanel(
                b,
                packedX,
                rows,
                order,
                0,
                0,
                transpose = !right,
                alpha = 1.0,
                destinationOffset = 0,
                workspace = null,
                structure = PackedPanelStructure.General,
                tileRows = tileRows,
            )
            var boundary = if (effectiveLower) order else 0
            while (if (effectiveLower) boundary > 0 else boundary < order) {
                val size = if (effectiveLower) {
                    val remainder = boundary % tileColumns
                    if (remainder == 0) tileColumns else remainder
                } else {
                    min(tileColumns, order - boundary)
                }
                val start = if (effectiveLower) boundary - size else boundary
                val end = start + size
                val innerStart = if (effectiveLower) end else 0
                val depth = if (effectiveLower) order - end else start
                val trianglePanel = start / tileColumns * order * tileColumns
                val triangleOffset = trianglePanel + start * tileColumns
                val updateOffset = trianglePanel + innerStart * tileColumns
                var rowStart = 0
                while (rowStart < rows) {
                    val validRows = min(tileRows, rows - rowStart)
                    val rowPanel = rowStart / tileRows * order * tileRows
                    val xOffset = rowPanel + start * tileRows
                    if (depth == 0) {
                        kernels.trsmTile(
                            validRows,
                            size,
                            packedTriangle,
                            triangleOffset,
                            effectiveLower,
                            unitDiag,
                            packedX,
                            xOffset,
                        )
                    } else {
                        kernels.gemmTrsmTile(
                            depth,
                            validRows,
                            size,
                            packedX,
                            rowPanel + innerStart * tileRows,
                            packedTriangle,
                            updateOffset,
                            packedTriangle,
                            triangleOffset,
                            effectiveLower,
                            unitDiag,
                            packedX,
                            xOffset,
                        )
                    }
                    rowStart += validRows
                }
                boundary = if (effectiveLower) start else end
            }
            writeLeftPanel(
                packedX,
                b,
                rows,
                order,
                0,
                0,
                0,
                transpose = !right,
                workspace = null,
                tileRows = tileRows,
            )
        }
    }
}
