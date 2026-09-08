package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.requireIndex
import com.eignex.koblas.requireNonNegativeShape
import com.eignex.koblas.requireShape
import kotlin.math.min

internal fun packedLeftSize(rows: Int, depth: Int, tileRows: Int): Int = packedPanelSize(rows, depth, tileRows, "left")

internal fun packedRightSize(depth: Int, columns: Int, tileColumns: Int): Int =
    packedPanelSize(columns, depth, tileColumns, "right")

private fun packedPanelSize(edges: Int, depth: Int, tile: Int, side: String): Int {
    requireNonNegativeShape(edges, depth)
    check(tile > 0) { "$side packed-panel tile must be positive, got $tile" }
    if (edges == 0 || depth == 0) return 0
    val groups = (edges.toLong() + tile - 1L) / tile
    val size = groups * depth * tile
    if (size > Int.MAX_VALUE) {
        throw DimensionMismatch("$side packed panel needs $size entries, more than one array can hold")
    }
    return size.toInt()
}

@Suppress("LongParameterList")
internal fun packLeftPanel(
    source: DenseMatrix,
    destination: DoubleArray,
    rows: Int,
    depth: Int,
    sourceRow: Int,
    sourceColumn: Int,
    transpose: Boolean,
    alpha: Double,
    destinationOffset: Int,
    workspace: Workspace?,
    structure: PackedPanelStructure,
    lower: Boolean = false,
    unitDiagonal: Boolean = false,
    tileRows: Int = PlatformKernels.gemmTileRows,
) {
    requirePackWindow(source, rows, depth, sourceRow, sourceColumn, transpose, structure)
    val size = packedLeftSize(rows, depth, tileRows)
    requireArrayWindow(destination, destinationOffset, size, "packed destination")
    withStableSource(source.data, destination, workspace) { stable ->
        var target = destinationOffset
        var row = 0
        while (row < rows) {
            val present = min(tileRows, rows - row)
            for (step in 0 until depth) {
                for (lane in 0 until present) {
                    val logicalRow = sourceRow + row + lane
                    val logicalColumn = sourceColumn + step
                    destination[target + lane] = if (isStructuralZero(
                            logicalRow,
                            logicalColumn,
                            transpose,
                            structure,
                            lower,
                        )
                    ) {
                        0.0
                    } else {
                        alpha * packedSourceValue(
                            stable,
                            source.rows,
                            logicalRow,
                            logicalColumn,
                            transpose,
                            structure,
                            lower,
                            unitDiagonal,
                        )
                    }
                }
                for (lane in present until tileRows) destination[target + lane] = 0.0
                target += tileRows
            }
            row += tileRows
        }
    }
}

@Suppress("LongParameterList")
internal fun packRightPanel(
    source: DenseMatrix,
    destination: DoubleArray,
    depth: Int,
    columns: Int,
    sourceRow: Int,
    sourceColumn: Int,
    transpose: Boolean,
    destinationOffset: Int,
    workspace: Workspace?,
    structure: PackedPanelStructure,
    lower: Boolean = false,
    unitDiagonal: Boolean = false,
    tileColumns: Int = PlatformKernels.gemmTileCols,
) {
    requirePackWindow(source, depth, columns, sourceRow, sourceColumn, transpose, structure)
    val size = packedRightSize(depth, columns, tileColumns)
    requireArrayWindow(destination, destinationOffset, size, "packed destination")
    withStableSource(source.data, destination, workspace) { stable ->
        var target = destinationOffset
        var column = 0
        while (column < columns) {
            val present = min(tileColumns, columns - column)
            for (step in 0 until depth) {
                for (lane in 0 until present) {
                    destination[target + lane] = packedSourceValue(
                        stable,
                        source.rows,
                        sourceRow + step,
                        sourceColumn + column + lane,
                        transpose,
                        structure,
                        lower,
                        unitDiagonal,
                    )
                }
                for (lane in present until tileColumns) destination[target + lane] = 0.0
                target += tileColumns
            }
            column += tileColumns
        }
    }
}

@Suppress("LongParameterList")
internal fun writeLeftPanel(
    source: DoubleArray,
    destination: DenseMatrix,
    rows: Int,
    depth: Int,
    sourceOffset: Int,
    destinationRow: Int,
    destinationColumn: Int,
    transpose: Boolean,
    workspace: Workspace?,
    tileRows: Int = PlatformKernels.gemmTileRows,
) {
    val size = packedLeftSize(rows, depth, tileRows)
    requireArrayWindow(source, sourceOffset, size, "packed source")
    requireWriteWindow(destination, rows, depth, destinationRow, destinationColumn, transpose)
    withStableSource(source, destination.data, workspace) { stable ->
        var packed = sourceOffset
        var row = 0
        while (row < rows) {
            val present = min(tileRows, rows - row)
            for (step in 0 until depth) {
                for (lane in 0 until present) {
                    writeLogical(
                        destination,
                        destinationRow,
                        destinationColumn,
                        row + lane,
                        step,
                        transpose,
                        stable[packed + lane],
                    )
                }
                packed += tileRows
            }
            row += tileRows
        }
    }
}

@Suppress("LongParameterList")
internal fun writeRightPanel(
    source: DoubleArray,
    destination: DenseMatrix,
    depth: Int,
    columns: Int,
    sourceOffset: Int,
    destinationRow: Int,
    destinationColumn: Int,
    transpose: Boolean,
    workspace: Workspace?,
    tileColumns: Int = PlatformKernels.gemmTileCols,
) {
    val size = packedRightSize(depth, columns, tileColumns)
    requireArrayWindow(source, sourceOffset, size, "packed source")
    requireWriteWindow(destination, depth, columns, destinationRow, destinationColumn, transpose)
    withStableSource(source, destination.data, workspace) { stable ->
        var packed = sourceOffset
        var column = 0
        while (column < columns) {
            val present = min(tileColumns, columns - column)
            for (step in 0 until depth) {
                for (lane in 0 until present) {
                    writeLogical(
                        destination,
                        destinationRow,
                        destinationColumn,
                        step,
                        column + lane,
                        transpose,
                        stable[packed + lane],
                    )
                }
                packed += tileColumns
            }
            column += tileColumns
        }
    }
}

internal fun clearLeftPanelPadding(panel: DoubleArray, rows: Int, depth: Int, panelOffset: Int, tileRows: Int) {
    val size = packedLeftSize(rows, depth, tileRows)
    requireArrayWindow(panel, panelOffset, size, "packed panel")
    val edge = rows % tileRows
    if (edge == 0 || depth == 0) return
    val edgePanel = panelOffset + (rows / tileRows) * depth * tileRows
    for (step in 0 until depth) {
        val group = edgePanel + step * tileRows
        panel.fill(0.0, group + edge, group + tileRows)
    }
}

internal fun clearRightPanelPadding(panel: DoubleArray, depth: Int, columns: Int, panelOffset: Int, tileColumns: Int) {
    val size = packedRightSize(depth, columns, tileColumns)
    requireArrayWindow(panel, panelOffset, size, "packed panel")
    val edge = columns % tileColumns
    if (edge == 0 || depth == 0) return
    val edgePanel = panelOffset + (columns / tileColumns) * depth * tileColumns
    for (step in 0 until depth) {
        val group = edgePanel + step * tileColumns
        panel.fill(0.0, group + edge, group + tileColumns)
    }
}

@Suppress("LongParameterList")
private fun requirePackWindow(
    source: DenseMatrix,
    logicalRows: Int,
    logicalColumns: Int,
    sourceRow: Int,
    sourceColumn: Int,
    transpose: Boolean,
    structure: PackedPanelStructure,
) {
    requireNonNegativeShape(logicalRows, logicalColumns)
    if (structure != PackedPanelStructure.General) {
        requireShape(source.rows == source.cols) {
            "structured panel source must be square; got ${source.rows}x${source.cols}"
        }
    }
    val availableRows = if (transpose) source.cols else source.rows
    val availableColumns = if (transpose) source.rows else source.cols
    requireMatrixWindow(sourceRow, sourceColumn, logicalRows, logicalColumns, availableRows, availableColumns)
}

private fun requireWriteWindow(
    destination: DenseMatrix,
    logicalRows: Int,
    logicalColumns: Int,
    destinationRow: Int,
    destinationColumn: Int,
    transpose: Boolean,
) {
    requireNonNegativeShape(logicalRows, logicalColumns)
    val writtenRows = if (transpose) logicalColumns else logicalRows
    val writtenColumns = if (transpose) logicalRows else logicalColumns
    requireMatrixWindow(
        destinationRow,
        destinationColumn,
        writtenRows,
        writtenColumns,
        destination.rows,
        destination.cols,
    )
}

private fun requireMatrixWindow(row: Int, column: Int, rows: Int, columns: Int, totalRows: Int, totalColumns: Int) {
    requireIndex(row >= 0 && column >= 0) { "matrix window starts at ($row, $column)" }
    requireIndex(row.toLong() + rows <= totalRows && column.toLong() + columns <= totalColumns) {
        "matrix window ($row, $column) + ${rows}x$columns exceeds ${totalRows}x$totalColumns"
    }
}

internal fun requireArrayWindow(array: DoubleArray, offset: Int, size: Int, what: String) {
    requireIndex(offset >= 0 && offset.toLong() + size <= array.size) {
        "$what [$offset, ${offset.toLong() + size}) exceeds array length ${array.size}"
    }
}

private fun packedSourceValue(
    source: DoubleArray,
    leadingDimension: Int,
    logicalRow: Int,
    logicalColumn: Int,
    transpose: Boolean,
    structure: PackedPanelStructure,
    lower: Boolean,
    unitDiagonal: Boolean,
): Double {
    val row = if (transpose) logicalColumn else logicalRow
    val column = if (transpose) logicalRow else logicalColumn
    return when (structure) {
        PackedPanelStructure.General -> source[row + column * leadingDimension]

        PackedPanelStructure.Symmetric -> {
            val stored = if (lower) row >= column else row <= column
            if (stored) {
                source[row + column * leadingDimension]
            } else {
                source[column + row * leadingDimension]
            }
        }

        PackedPanelStructure.Triangular -> when {
            row == column && unitDiagonal -> 1.0
            if (lower) row >= column else row <= column -> source[row + column * leadingDimension]
            else -> 0.0
        }
    }
}

private fun isStructuralZero(
    logicalRow: Int,
    logicalColumn: Int,
    transpose: Boolean,
    structure: PackedPanelStructure,
    lower: Boolean,
): Boolean {
    if (structure != PackedPanelStructure.Triangular) return false
    val row = if (transpose) logicalColumn else logicalRow
    val column = if (transpose) logicalRow else logicalColumn
    return if (lower) row < column else row > column
}

private fun writeLogical(
    destination: DenseMatrix,
    destinationRow: Int,
    destinationColumn: Int,
    logicalRow: Int,
    logicalColumn: Int,
    transpose: Boolean,
    value: Double,
) {
    val row = destinationRow + if (transpose) logicalColumn else logicalRow
    val column = destinationColumn + if (transpose) logicalRow else logicalColumn
    destination.data[row + column * destination.rows] = value
}

internal inline fun withStableSource(
    source: DoubleArray,
    destination: DoubleArray,
    workspace: Workspace?,
    block: (DoubleArray) -> Unit,
) {
    if (source !== destination) {
        block(source)
        return
    }
    workspace.borrow(source.size) { staged ->
        source.copyInto(staged)
        block(staged)
    }
}
