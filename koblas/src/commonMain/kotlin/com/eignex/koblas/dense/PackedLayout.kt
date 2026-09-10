@file:Suppress("MatchingDeclarationName") // packed structure and its layout operations are one unit

package com.eignex.koblas.dense

import kotlin.math.min

internal enum class PackedPanelStructure { General, Symmetric, Triangular }

/**
 * Trusted array-level layout leaves for the padded panels consumed by [PackedKernels]. Callers provide valid,
 * non-overlapping windows and a positive tile shape. Checked APIs stage aliases before entering these leaves;
 * internal level-3 orchestration owns distinct scratch panels already.
 */

@Suppress("LongParameterList")
internal fun packLeftLayout(
    source: DoubleArray,
    leadingDimension: Int,
    destination: DoubleArray,
    destinationOffset: Int,
    rows: Int,
    depth: Int,
    sourceRow: Int,
    sourceColumn: Int,
    transpose: Boolean,
    alpha: Double,
    structure: PackedPanelStructure,
    lower: Boolean,
    unitDiagonal: Boolean,
    tileRows: Int,
) {
    var target = destinationOffset
    var row = 0
    while (row < rows) {
        val present = min(tileRows, rows - row)
        var step = 0
        while (step < depth) {
            val logicalRow = sourceRow + row
            val logicalColumn = sourceColumn + step
            var lane = 0
            when (structure) {
                PackedPanelStructure.General -> while (lane < present) {
                    val i = logicalRow + lane
                    val value = if (transpose) {
                        source[logicalColumn + i * leadingDimension]
                    } else {
                        source[i + logicalColumn * leadingDimension]
                    }
                    destination[target + lane] = alpha * value
                    lane++
                }

                PackedPanelStructure.Symmetric -> while (lane < present) {
                    val i = logicalRow + lane
                    val stored = if (lower) i >= logicalColumn else i <= logicalColumn
                    val value = if (stored) {
                        source[i + logicalColumn * leadingDimension]
                    } else {
                        source[logicalColumn + i * leadingDimension]
                    }
                    destination[target + lane] = alpha * value
                    lane++
                }

                PackedPanelStructure.Triangular -> while (lane < present) {
                    val logicalLane = logicalRow + lane
                    val i = if (transpose) logicalColumn else logicalLane
                    val j = if (transpose) logicalLane else logicalColumn
                    val stored = if (lower) i >= j else i <= j
                    destination[target + lane] = when {
                        !stored -> 0.0
                        i == j && unitDiagonal -> alpha * 1.0
                        else -> alpha * source[i + j * leadingDimension]
                    }
                    lane++
                }
            }
            destination.fill(0.0, target + present, target + tileRows)
            target += tileRows
            step++
        }
        row += tileRows
    }
}

@Suppress("LongParameterList")
internal fun packRightLayout(
    source: DoubleArray,
    leadingDimension: Int,
    destination: DoubleArray,
    destinationOffset: Int,
    depth: Int,
    columns: Int,
    sourceRow: Int,
    sourceColumn: Int,
    transpose: Boolean,
    structure: PackedPanelStructure,
    lower: Boolean,
    unitDiagonal: Boolean,
    tileColumns: Int,
) {
    var target = destinationOffset
    var column = 0
    while (column < columns) {
        val present = min(tileColumns, columns - column)
        var step = 0
        while (step < depth) {
            val logicalRow = sourceRow + step
            val logicalColumn = sourceColumn + column
            var lane = 0
            when (structure) {
                PackedPanelStructure.General -> while (lane < present) {
                    val j = logicalColumn + lane
                    destination[target + lane] = if (transpose) {
                        source[j + logicalRow * leadingDimension]
                    } else {
                        source[logicalRow + j * leadingDimension]
                    }
                    lane++
                }

                PackedPanelStructure.Symmetric -> while (lane < present) {
                    val j = logicalColumn + lane
                    val stored = if (lower) logicalRow >= j else logicalRow <= j
                    destination[target + lane] = if (stored) {
                        source[logicalRow + j * leadingDimension]
                    } else {
                        source[j + logicalRow * leadingDimension]
                    }
                    lane++
                }

                PackedPanelStructure.Triangular -> while (lane < present) {
                    val logicalLane = logicalColumn + lane
                    val i = if (transpose) logicalLane else logicalRow
                    val j = if (transpose) logicalRow else logicalLane
                    val stored = if (lower) i >= j else i <= j
                    destination[target + lane] = when {
                        !stored -> 0.0
                        i == j && unitDiagonal -> 1.0
                        else -> source[i + j * leadingDimension]
                    }
                    lane++
                }
            }
            destination.fill(0.0, target + present, target + tileColumns)
            target += tileColumns
            step++
        }
        column += tileColumns
    }
}

@Suppress("LongParameterList")
internal fun writeLeftLayout(
    source: DoubleArray,
    sourceOffset: Int,
    destination: DoubleArray,
    leadingDimension: Int,
    rows: Int,
    depth: Int,
    destinationRow: Int,
    destinationColumn: Int,
    transpose: Boolean,
    tileRows: Int,
) {
    var packed = sourceOffset
    var row = 0
    while (row < rows) {
        val present = min(tileRows, rows - row)
        var step = 0
        while (step < depth) {
            var lane = 0
            while (lane < present) {
                val logicalRow = row + lane
                val targetRow = destinationRow + if (transpose) step else logicalRow
                val targetColumn = destinationColumn + if (transpose) logicalRow else step
                destination[targetRow + targetColumn * leadingDimension] = source[packed + lane]
                lane++
            }
            packed += tileRows
            step++
        }
        row += tileRows
    }
}

@Suppress("LongParameterList")
internal fun writeRightLayout(
    source: DoubleArray,
    sourceOffset: Int,
    destination: DoubleArray,
    leadingDimension: Int,
    depth: Int,
    columns: Int,
    destinationRow: Int,
    destinationColumn: Int,
    transpose: Boolean,
    tileColumns: Int,
) {
    var packed = sourceOffset
    var column = 0
    while (column < columns) {
        val present = min(tileColumns, columns - column)
        var step = 0
        while (step < depth) {
            var lane = 0
            while (lane < present) {
                val logicalColumn = column + lane
                val targetRow = destinationRow + if (transpose) logicalColumn else step
                val targetColumn = destinationColumn + if (transpose) step else logicalColumn
                destination[targetRow + targetColumn * leadingDimension] = source[packed + lane]
                lane++
            }
            packed += tileColumns
            step++
        }
        column += tileColumns
    }
}

internal fun clearLeftLayoutPadding(panel: DoubleArray, rows: Int, depth: Int, panelOffset: Int, tileRows: Int) {
    val edge = rows % tileRows
    if (edge == 0 || depth == 0) return
    val edgePanel = panelOffset + rows / tileRows * depth * tileRows
    var step = 0
    while (step < depth) {
        val group = edgePanel + step * tileRows
        panel.fill(0.0, group + edge, group + tileRows)
        step++
    }
}

internal fun clearRightLayoutPadding(
    panel: DoubleArray,
    depth: Int,
    columns: Int,
    panelOffset: Int,
    tileColumns: Int,
) {
    val edge = columns % tileColumns
    if (edge == 0 || depth == 0) return
    val edgePanel = panelOffset + columns / tileColumns * depth * tileColumns
    var step = 0
    while (step < depth) {
        val group = edgePanel + step * tileColumns
        panel.fill(0.0, group + edge, group + tileColumns)
        step++
    }
}
