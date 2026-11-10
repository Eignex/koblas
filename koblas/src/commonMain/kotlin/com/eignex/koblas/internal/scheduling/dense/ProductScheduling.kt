@file:Suppress("LongParameterList") // a product carries both operands, both transpose flags and three extents

package com.eignex.koblas.internal.scheduling.dense

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.operations.applyBeta
import com.eignex.koblas.internal.scheduling.forEachPanel
import com.eignex.koblas.internal.storage.borrowOptional

/*
 * Shared scheduling for `C = alpha * op(A) * op(B) + beta * C`.
 * Large windows pack cache blocks for [EngineKernels.productBlock]; small or thin ones use Level 2 panels.
 * Beta applies once per destination window, on the first depth block or unpacked column writeback.
 * Offsets and leading dimensions let structured routines pass strips and blocks without copying.
 */

/** Destination rows one cache block covers. */
internal const val PRODUCT_BLOCK_ROWS: Int = 128

/** Destination columns one cache block covers. */
internal const val PRODUCT_BLOCK_COLUMNS: Int = 256

/**
 * Depth accumulated before touching the destination again. Deeper blocks trade fewer destination passes
 * for larger packed panels. Cache geometry is independent of the backend's register tile.
 */
internal const val PRODUCT_BLOCK_DEPTH: Int = 128

/**
 * The writable destination region. The opposite triangle is neither read nor written:
 * blocks outside it are skipped, and blocks crossing its diagonal merge only selected entries.
 */
internal enum class OutputTriangle {
    /** Every entry of the destination window is written. */
    Full,

    /** Only entries on or below the diagonal of the destination window are written. */
    Lower,

    /** Only entries on or above the diagonal of the destination window are written. */
    Upper,
}

/**
 * Executes the path selected by this window's extents, including strips passed by structured routines.
 * Packing must amortize both operand copies; other windows use panels over the original storage.
 */
@Suppress("NOTHING_TO_INLINE") // Keep transient region views inside their caller for JVM escape analysis.
internal inline fun productWindow(
    engine: KoblasEngine,
    alpha: Double,
    a: DenseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrix,
    selected: OutputTriangle,
    arrays: Scratch?,
) {
    val m = c.rows
    val n = c.cols
    val k = if (transposeA) a.rows else a.cols
    if (m <= 0 || n <= 0) return
    if (k <= 0) {
        // Neither path has a depth step to apply beta on, so an empty product scales the selected entries here.
        for (j in 0 until n) {
            val at = c.offset + selectedRow(j, m, selected) + j * c.leadingDimension
            applyBeta(engine, c.values, at, selectedRows(j, m, selected), beta)
        }
        return
    }
    // A selected triangle uses the rectangular crossover over its actual extents: reducing the depth to stand
    // for discarded triangle work would distort what a backend decides from.
    val kernels = engine.productProvider
    if (kernels.packsProduct(m, n, k)) {
        blockedProduct(
            kernels, alpha, a.values, a.offset, a.leadingDimension, transposeA,
            b.values, b.offset, b.leadingDimension, transposeB,
            beta, c.values, c.offset, c.leadingDimension, m, n, k, selected, arrays,
        )
    } else {
        directProduct(
            engine.panelProvider, alpha, a.values, a.offset, a.leadingDimension, transposeA,
            b.values, b.offset, b.leadingDimension, transposeB,
            beta, c.values, c.offset, c.leadingDimension, m, n, k, selected, arrays,
        )
    }
}

/** Transient cache-block packing, with a tile-sized merge buffer for a selected triangle. */
internal fun blockedProduct(
    kernels: EngineKernels,
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    transposeA: Boolean,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    transposeB: Boolean,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    m: Int,
    n: Int,
    k: Int,
    selected: OutputTriangle,
    arrays: Scratch?,
) {
    val blockRows = productBlockRows(kernels, m)
    val blockColumns = productBlockColumns(kernels, n)
    val blockDepth = productBlockDepth(k)
    val leftSize = scratchCapacity(blockRows * blockDepth)
    val rightSize = scratchCapacity(blockColumns * blockDepth)
    val edgeSize = if (selected == OutputTriangle.Full) 0 else kernels.tileRows * kernels.tileColumns
    arrays.borrowOptional(leftSize) { left ->
        arrays.borrowOptional(rightSize) { right ->
            arrays.borrowOptional(edgeSize) { edge ->
                blockedProductCore(
                    kernels, alpha, a, aOffset, lda, transposeA, left,
                    b, bOffset, ldb, transposeB, right,
                    beta, c, cOffset, ldc, m, n, k, blockRows, blockColumns, blockDepth, selected, edge,
                )
            }
        }
    }
}

/**
 * Column, depth, then row blocks reuse each packed right panel across a destination strip
 * while keeping the live destination working set within one block.
 */
internal inline fun forEachProductBlock(
    m: Int,
    n: Int,
    k: Int,
    blockRows: Int,
    blockColumns: Int,
    blockDepth: Int,
    action: (row: Int, rowCount: Int, column: Int, columnCount: Int, step: Int, depth: Int) -> Unit,
) {
    var column = 0
    while (column < n) {
        val columnCount = if (blockColumns < n - column) blockColumns else n - column
        var step = 0
        while (step < k) {
            val depth = if (blockDepth < k - step) blockDepth else k - step
            var row = 0
            while (row < m) {
                val rowCount = if (blockRows < m - row) blockRows else m - row
                action(row, rowCount, column, columnCount, step, depth)
                row += rowCount
            }
            step += depth
        }
        column += columnCount
    }
}

/** The block extents a product of these logical extents is scheduled with, for [forEachProductBlock]. */
internal fun productBlockRows(kernels: EngineKernels, m: Int): Int = blockExtent(
    PRODUCT_BLOCK_ROWS,
    kernels.tileRows,
    m,
)

/** The destination columns one block covers, rounded to whole tiles. */
internal fun productBlockColumns(kernels: EngineKernels, n: Int): Int =
    blockExtent(PRODUCT_BLOCK_COLUMNS, kernels.tileColumns, n)

/** The shared-dimension steps one block accumulates, which needs no rounding. */
internal fun productBlockDepth(k: Int): Int = if (PRODUCT_BLOCK_DEPTH < k) PRODUCT_BLOCK_DEPTH else k

/** Whether the destination window lies wholly outside the selected triangle. */
internal fun outsideTriangle(
    row: Int,
    rowCount: Int,
    column: Int,
    columnCount: Int,
    selected: OutputTriangle,
): Boolean = when (selected) {
    OutputTriangle.Full -> false
    OutputTriangle.Lower -> row + rowCount - 1 < column
    OutputTriangle.Upper -> row > column + columnCount - 1
}

/** Whether every entry of a destination window is selected, so the window is written as an ordinary one. */
internal fun insideTriangle(
    row: Int,
    rowCount: Int,
    column: Int,
    columnCount: Int,
    selected: OutputTriangle,
): Boolean = when (selected) {
    OutputTriangle.Full -> true
    OutputTriangle.Lower -> row >= column + columnCount - 1
    OutputTriangle.Upper -> row + rowCount - 1 <= column
}

/**
 * Tiles crossing the diagonal merge only selected entries; `whole` tiles write directly.
 * Tiles wholly outside the selected triangle are skipped.
 */
internal inline fun forEachSelectedTile(
    kernels: EngineKernels,
    row: Int,
    rowCount: Int,
    column: Int,
    columnCount: Int,
    selected: OutputTriangle,
    action: (tileRow: Int, rows: Int, tileColumn: Int, columns: Int, whole: Boolean) -> Unit,
) {
    val tileRows = kernels.tileRows
    val tileColumns = kernels.tileColumns
    var tileColumn = 0
    while (tileColumn < columnCount) {
        val columns = if (tileColumns < columnCount - tileColumn) tileColumns else columnCount - tileColumn
        var tileRow = 0
        while (tileRow < rowCount) {
            val rows = if (tileRows < rowCount - tileRow) tileRows else rowCount - tileRow
            val at = row + tileRow
            val from = column + tileColumn
            if (!outsideTriangle(at, rows, from, columns, selected)) {
                action(tileRow, rows, tileColumn, columns, insideTriangle(at, rows, from, columns, selected))
            }
            tileRow += tileRows
        }
        tileColumn += tileColumns
    }
}

/** The block traversal itself, over panels packed into [left] and [right]. */
@Suppress("CyclomaticComplexMethod") // the traversal, its lazy packing and the three positions of a block
private fun blockedProductCore(
    kernels: EngineKernels,
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    transposeA: Boolean,
    left: DoubleArray,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    transposeB: Boolean,
    right: DoubleArray,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    m: Int,
    n: Int,
    k: Int,
    blockRows: Int,
    blockColumns: Int,
    blockDepth: Int,
    selected: OutputTriangle,
    edge: DoubleArray,
) {
    val tileRows = kernels.tileRows
    val tileColumns = kernels.tileColumns
    var packedColumn = -1
    var packedStep = -1
    var rightStride = 0
    forEachProductBlock(
        m,
        n,
        k,
        blockRows,
        blockColumns,
        blockDepth,
    ) { row, rowCount, column, columnCount, step, depth ->
        // Skip blocks outside the selected triangle without packing operands or touching the destination.
        if (!outsideTriangle(row, rowCount, column, columnCount, selected)) {
            // Reuse the right panel across row blocks for the same column and depth block.
            if (column != packedColumn || step != packedStep) {
                packLeftPanel(b, bOffset, ldb, !transposeB, column, step, columnCount, depth, right, 0, tileColumns)
                rightStride = depth * tileColumns
                packedColumn = column
                packedStep = step
            }
            packLeftPanel(a, aOffset, lda, transposeA, row, step, rowCount, depth, left, 0, tileRows)
            val leftStride = depth * tileRows
            // Apply beta only in the first depth block; later blocks accumulate.
            val blockBeta = if (step == 0) beta else 1.0
            if (insideTriangle(row, rowCount, column, columnCount, selected)) {
                kernels.productBlock(
                    alpha, left, 0, leftStride, right, 0, rightStride,
                    rowCount, columnCount, depth, blockBeta, c, cOffset + row + column * ldc, ldc,
                )
            } else {
                selectedBlock(
                    kernels, alpha, left, leftStride, right, rightStride,
                    row, rowCount, column, columnCount, depth, blockBeta, c, cOffset, ldc, selected, edge,
                )
            }
        }
    }
}

/**
 * Writes a diagonal-crossing block using its already packed panels. Whole tiles write directly;
 * crossing tiles accumulate into [edge] and merge selected entries, applying beta once.
 * Merging per register tile bounds discarded work to part of a tile instead of part of a cache block.
 */
private fun selectedBlock(
    kernels: EngineKernels,
    alpha: Double,
    leftPanel: DoubleArray,
    leftStride: Int,
    rightPanel: DoubleArray,
    rightStride: Int,
    row: Int,
    rowCount: Int,
    column: Int,
    columnCount: Int,
    depth: Int,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    selected: OutputTriangle,
    edge: DoubleArray,
) {
    val tileRows = kernels.tileRows
    forEachSelectedTile(
        kernels,
        row,
        rowCount,
        column,
        columnCount,
        selected,
    ) { tileRow, rows, tileColumn, columns, whole ->
        val leftTile = tileRow / tileRows * leftStride
        val rightTile = tileColumn / kernels.tileColumns * rightStride
        val at = row + tileRow
        val from = column + tileColumn
        val target = cOffset + at + from * ldc
        if (whole) {
            kernels.productBlock(
                alpha, leftPanel, leftTile, leftStride, rightPanel, rightTile, rightStride,
                rows, columns, depth, beta, c, target, ldc,
            )
        } else {
            kernels.productBlock(
                alpha, leftPanel, leftTile, leftStride, rightPanel, rightTile, rightStride,
                rows, columns, depth, 0.0, edge, 0, tileRows,
            )
            mergeSelected(edge, tileRows, rows, columns, beta, c, target, ldc, at, from, selected)
        }
    }
}

/**
 * The selected entries of an accumulated tile, added into the destination with its multiplier spent once.
 *
 * A zero multiplier overwrites without reading, for the reason [scaleProductWindow] gives.
 */
private fun mergeSelected(
    edge: DoubleArray,
    tileRows: Int,
    rows: Int,
    columns: Int,
    beta: Double,
    c: DoubleArray,
    at: Int,
    ldc: Int,
    rowOrigin: Int,
    columnOrigin: Int,
    selected: OutputTriangle,
) {
    for (column in 0 until columns) {
        val target = at + column * ldc
        val source = column * tileRows
        for (row in 0 until rows) {
            val keep = if (selected == OutputTriangle.Lower) {
                rowOrigin + row >= columnOrigin + column
            } else {
                rowOrigin + row <= columnOrigin + column
            }
            if (!keep) continue
            val value = edge[source + row]
            c[target + row] = when (beta) {
                0.0 -> value
                1.0 -> c[target + row] + value
                else -> value + beta * c[target + row]
            }
        }
    }
}

/**
 * Unpacked Level 2 panels, shortened to the selected destination rows so no other entries are read or written.
 * Untransposed panels use a bounded accumulation column, measured faster than direct destination updates;
 * transposed panels apply both multipliers on writeback. Backend rounding may differ.
 * [gathersCoefficients] gathers strided columns of `op(B)` when vector reductions amortize the copy.
 */
internal fun directProduct(
    panels: EngineKernels,
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    transposeA: Boolean,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    transposeB: Boolean,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    m: Int,
    n: Int,
    k: Int,
    selected: OutputTriangle,
    arrays: Scratch?,
) {
    val coefficientStride = if (transposeB) ldb else 1
    if (transposeA) {
        reducedProduct(
            panels, alpha, a, aOffset, lda, b, bOffset, ldb, transposeB, coefficientStride,
            beta, c, cOffset, ldc, m, n, k, selected, arrays,
        )
        return
    }
    val chunk = directColumnBlock(m)
    val group = directPanelGroup(panels, m, k, transposeA = false)
    arrays.borrowOptional(scratchCapacity(chunk)) { accumulated ->
        forEachDirectProductPanel(m, n, k, selected, false, group) { column, first, step, height, width ->
            if (step == 0) accumulated.fill(0.0, 0, height)
            val coefficients = bOffset + if (transposeB) column else column * ldb
            panels.columnUpdate(
                1.0, a, aOffset + first + step * lda, lda,
                b, coefficients + step * coefficientStride, coefficientStride,
                height, width, accumulated, 0, 1,
            )
            if (step + width == k) {
                writeScaledColumn(alpha, accumulated, beta, c, cOffset + first + column * ldc, height)
            }
        }
    }
}

/** Bounds the accumulating column independently of the call order and backend panel grouping. */
internal fun directColumnBlock(m: Int): Int = if (m < PRODUCT_BLOCK_ROWS) m else PRODUCT_BLOCK_ROWS

/**
 * How many columns of the unpacked path's panel the backend recommends handing over at a time.
 *
 * A transposed left operand reduces down stored columns and an untransposed one accumulates them into the
 * destination, which are different panels over different shapes, so each is grouped over the shape it cuts.
 */
internal fun directPanelGroup(panels: EngineKernels, m: Int, k: Int, transposeA: Boolean): Int = if (transposeA) {
    panels.executionGroup(PanelWork.MultiDot, k, m)
} else {
    panels.executionGroup(PanelWork.ColumnUpdate, directColumnBlock(m), k)
}

/** The transposed-left half of [directProduct], with or without the gathered coefficient column. */
private fun reducedProduct(
    panels: EngineKernels,
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    transposeB: Boolean,
    coefficientStride: Int,
    beta: Double,
    c: DoubleArray,
    cOffset: Int,
    ldc: Int,
    m: Int,
    n: Int,
    k: Int,
    selected: OutputTriangle,
    arrays: Scratch?,
) {
    val group = directPanelGroup(panels, m, k, transposeA = true)
    val gathers = gathersCoefficients(panels, m, k, coefficientStride != 1)
    arrays.borrowOptional(if (gathers) scratchCapacity(k) else 0) { gathered ->
        forEachDirectProductPanel(m, n, k, selected, true, group) { column, first, _, height, width ->
            val coefficients = bOffset + if (transposeB) column else column * ldb
            if (gathers && first == selectedRow(column, m, selected)) {
                for (p in 0 until k) gathered[p] = b[coefficients + p * coefficientStride]
            }
            panels.multiDot(
                alpha, a, aOffset + first * lda, lda,
                if (gathers) gathered else b, if (gathers) 0 else coefficients, if (gathers) 1 else coefficientStride,
                height, width, beta, c, cOffset + first + column * ldc, 1,
            )
        }
    }
}

/** The first destination row column [j] may be written at, which the selected triangle decides. */
internal fun selectedRow(j: Int, m: Int, selected: OutputTriangle): Int =
    if (selected == OutputTriangle.Lower) if (j < m) j else m else 0

/** How many destination rows column [j] may be written at, counted from [selectedRow]. */
internal fun selectedRows(j: Int, m: Int, selected: OutputTriangle): Int = when (selected) {
    OutputTriangle.Full -> m
    OutputTriangle.Lower -> if (j < m) m - j else 0
    OutputTriangle.Upper -> if (j + 1 < m) j + 1 else m
}

/**
 * Gather a strided coefficient column only if it enables a different reduction body and enough destination
 * rows reuse it to amortize the copy. [DIRECT_GATHER_MINIMUM_ROWS] reflects local measurements:
 * eight rows clearly won, while four varied between captures. Other hosts need their own evidence.
 */
internal fun gathersCoefficients(panels: EngineKernels, rows: Int, depth: Int, strided: Boolean): Boolean {
    if (!strided || rows < DIRECT_GATHER_MINIMUM_ROWS || depth <= 0) return false
    val group = panels.executionGroup(PanelWork.MultiDot, depth, rows)
    return panels.implementationFor(PanelWork.MultiDot, depth, group, contiguous = true) !=
        panels.implementationFor(PanelWork.MultiDot, depth, group, contiguous = false)
}

/** Destination rows from which a gathered coefficient column is measurably worth the pass it costs. */
internal const val DIRECT_GATHER_MINIMUM_ROWS: Int = 8

/**
 * `c = alpha · accumulated + beta · c` over one destination column, where a zero [beta] overwrites without
 * reading what is there, as the destination's contract requires rather than as an arithmetic shortcut.
 */
private fun writeScaledColumn(
    alpha: Double,
    accumulated: DoubleArray,
    beta: Double,
    c: DoubleArray,
    at: Int,
    rows: Int,
) {
    if (beta == 0.0) {
        for (i in 0 until rows) c[at + i] = alpha * accumulated[i]
    } else {
        for (i in 0 until rows) c[at + i] = alpha * accumulated[i] + beta * c[at + i]
    }
}

/**
 * Round scratch lengths so neighboring windows reuse arrays loans. Schedule blocks bound the
 * accumulating column and packed panels; rounding retains at most twice the requested storage.
 * Oversized lengths pass through.
 */
internal fun scratchCapacity(size: Int): Int {
    if (size <= 1 || size > LARGEST_ROUNDED_SCRATCH) return size
    return (size - 1).takeHighestOneBit() shl 1
}

/** The largest request rounding applies to; above it the next power of two is not an array length. */
private const val LARGEST_ROUNDED_SCRATCH: Int = 1 shl 30

/**
 * Round [preferred] to whole tiles, capped by the extent. A one-tile floor prevents zero-step traversal;
 * all blocks except the tail start on tile boundaries.
 */
internal fun blockExtent(preferred: Int, tile: Int, extent: Int): Int {
    val whole = ((extent.toLong() + tile - 1) / tile * tile).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
    val block = preferred / tile * tile
    val atLeastOne = if (block < tile) tile else block
    return if (atLeastOne < whole) atLeastOne else whole
}

/** Direct arithmetic windows, including selected support and tails. */
internal inline fun forEachDirectProductPanel(
    m: Int,
    n: Int,
    k: Int,
    selected: OutputTriangle,
    transposeA: Boolean,
    group: Int,
    action: (column: Int, first: Int, step: Int, rows: Int, width: Int) -> Unit,
) {
    val chunk = directColumnBlock(m)
    for (column in 0 until n) {
        val first = selectedRow(column, m, selected)
        val rows = selectedRows(column, m, selected)
        if (transposeA) {
            forEachPanel(rows, group) { start, width -> action(column, first + start, 0, k, width) }
        } else {
            var offset = 0
            while (offset < rows) {
                val height = minOf(chunk, rows - offset)
                forEachPanel(k, group) { step, width -> action(column, first + offset, step, height, width) }
                offset += height
            }
        }
    }
}

/** The part of a square destination a selected-triangle routine writes. */
internal fun selectedTriangle(lower: Boolean): OutputTriangle =
    if (lower) OutputTriangle.Lower else OutputTriangle.Upper
