package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.requireIndex
import com.eignex.koblas.requireNonNegativeShape
import com.eignex.koblas.requireShape

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
    tileRows: Int = platformPackedKernels.gemmTileRows,
) {
    requirePackWindow(source, rows, depth, sourceRow, sourceColumn, transpose, structure)
    val size = packedLeftSize(rows, depth, tileRows)
    requireArrayWindow(destination, destinationOffset, size, "packed destination")
    withStableSource(source.data, destination, workspace) { stable ->
        packLeftLayout(
            stable, source.rows, destination, destinationOffset, rows, depth, sourceRow, sourceColumn,
            transpose, alpha, structure, lower, unitDiagonal, tileRows,
        )
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
    tileColumns: Int = platformPackedKernels.gemmTileCols,
) {
    requirePackWindow(source, depth, columns, sourceRow, sourceColumn, transpose, structure)
    val size = packedRightSize(depth, columns, tileColumns)
    requireArrayWindow(destination, destinationOffset, size, "packed destination")
    withStableSource(source.data, destination, workspace) { stable ->
        packRightLayout(
            stable, source.rows, destination, destinationOffset, depth, columns, sourceRow, sourceColumn,
            transpose, structure, lower, unitDiagonal, tileColumns,
        )
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
    tileRows: Int = platformPackedKernels.gemmTileRows,
) {
    val size = packedLeftSize(rows, depth, tileRows)
    requireArrayWindow(source, sourceOffset, size, "packed source")
    requireWriteWindow(destination, rows, depth, destinationRow, destinationColumn, transpose)
    withStableSource(source, destination.data, workspace) { stable ->
        writeLeftLayout(
            stable, sourceOffset, destination.data, destination.rows, rows, depth,
            destinationRow, destinationColumn, transpose, tileRows,
        )
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
    tileColumns: Int = platformPackedKernels.gemmTileCols,
) {
    val size = packedRightSize(depth, columns, tileColumns)
    requireArrayWindow(source, sourceOffset, size, "packed source")
    requireWriteWindow(destination, depth, columns, destinationRow, destinationColumn, transpose)
    withStableSource(source, destination.data, workspace) { stable ->
        writeRightLayout(
            stable, sourceOffset, destination.data, destination.rows, depth, columns,
            destinationRow, destinationColumn, transpose, tileColumns,
        )
    }
}

internal fun clearLeftPanelPadding(panel: DoubleArray, rows: Int, depth: Int, panelOffset: Int, tileRows: Int) {
    val size = packedLeftSize(rows, depth, tileRows)
    requireArrayWindow(panel, panelOffset, size, "packed panel")
    clearLeftLayoutPadding(panel, rows, depth, panelOffset, tileRows)
}

internal fun clearRightPanelPadding(panel: DoubleArray, depth: Int, columns: Int, panelOffset: Int, tileColumns: Int) {
    val size = packedRightSize(depth, columns, tileColumns)
    requireArrayWindow(panel, panelOffset, size, "packed panel")
    clearRightLayoutPadding(panel, depth, columns, panelOffset, tileColumns)
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
