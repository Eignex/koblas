@file:Suppress("LongParameterList") // a product carries both operands, both transpose flags and three extents

package com.eignex.koblas.internal.dense

import com.eignex.koblas.*
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/*
 * Shared scheduling for `C = alpha * op(A) * op(B) + beta * C`.
 *
 * Large products use cache blocks packed for [DenseProductKernels.productBlock]; small or thin products
 * use Level 2 panels over the original storage, gathering strided coefficients when profitable.
 * The schedule owns traversal and applies beta once per output window: in the first depth block for packed
 * products, or when an unpacked column is written. Backends own the arithmetic within those windows.
 *
 * Array windows carry offsets and leading dimensions so structured routines can pass strips and blocks
 * without copying. An origin at logical (row, column) is `row + column * lda` in stored orientation and
 * `column + row * lda` when transposed.
 */

/** Destination rows one cache block covers. */
internal const val PRODUCT_BLOCK_ROWS: Int = 128

/** Destination columns one cache block covers. */
internal const val PRODUCT_BLOCK_COLUMNS: Int = 256

/**
 * Steps of the shared dimension a cache block accumulates before the destination is touched again.
 *
 * This sets the working set, since both packed panels are as deep as it: a block holds
 * [PRODUCT_BLOCK_ROWS] by this and this by [PRODUCT_BLOCK_COLUMNS] doubles at once. A destination tile stays
 * in registers for the whole of it, so a deeper block means fewer passes over the destination and a larger
 * pair of panels to keep resident. These three are the shared scheduling's and are independent of the
 * backend's register tile, which divides a block rather than sizing it.
 */
internal const val PRODUCT_BLOCK_DEPTH: Int = 128

private val NO_PANEL = DoubleArray(0)

/**
 * Optional [borrow] that skips zero-length loans so unused scratch does not consume workspace retention.
 */
internal inline fun <T> Workspace?.borrowOptional(size: Int, block: (DoubleArray) -> T): T =
    if (size == 0) block(NO_PANEL) else borrow(size, block)

/**
 * Which part of a square destination a product is allowed to write.
 *
 * [Full] is an ordinary rectangular product. The other two are the rank-k and triangle-selected routines,
 * whose contract is that the opposite triangle is neither read nor written, so a block lying wholly in it is
 * never scheduled and a block straddling the diagonal reaches the destination one selected entry at a time.
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
 * Executes the route selected by this window's extents, including strips passed by structured routines.
 * Packing must amortize both operand copies; other windows use panels over the original storage.
 */
internal fun productWindow(
    kernels: DenseProductKernels,
    panels: DensePanelKernels,
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
    workspace: Workspace?,
) {
    if (m <= 0 || n <= 0) return
    if (packsWindow(kernels, m, n, k, selected)) {
        blockedProduct(
            kernels, alpha, a, aOffset, lda, transposeA, b, bOffset, ldb, transposeB,
            beta, c, cOffset, ldc, m, n, k, selected, workspace,
        )
    } else {
        directProduct(
            panels, alpha, a, aOffset, lda, transposeA, b, bOffset, ldb, transposeB,
            beta, c, cOffset, ldc, m, n, k, selected, workspace,
        )
    }
}

/**
 * Whether a window is packed, using its actual extents even for a selected triangle.
 *
 * Backends may decide from depth, minimum dimensions or tails, so reducing depth to represent discarded
 * triangle work would distort their inputs. Selected triangles use the rectangular crossover; their
 * lower arithmetic per copied value may merit a higher threshold, but [packsProductByWork] records no
 * reliable secondary criterion. Routing and execution share this decision.
 */
@Suppress("UNUSED_PARAMETER") // the selected triangle is part of the question even where the answer ignores it
internal fun packsWindow(kernels: DenseProductKernels, m: Int, n: Int, k: Int, selected: OutputTriangle): Boolean =
    kernels.packsProduct(m, n, k)

/** Transient cache-block packing, with a tile-sized merge buffer for a selected triangle. */
internal fun blockedProduct(
    kernels: DenseProductKernels,
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
    workspace: Workspace?,
) {
    val blockRows = productBlockRows(kernels, m)
    val blockColumns = productBlockColumns(kernels, n)
    val blockDepth = productBlockDepth(k)
    val leftSize = scratchCapacity(blockRows * blockDepth)
    val rightSize = scratchCapacity(blockColumns * blockDepth)
    val edgeSize = if (selected == OutputTriangle.Full) 0 else kernels.tileRows * kernels.tileColumns
    workspace.borrowOptional(leftSize) { left ->
        workspace.borrowOptional(rightSize) { right ->
            workspace.borrowOptional(edgeSize) { edge ->
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
 * The blocks a product of these extents is cut into, in the order the traversal reaches them.
 *
 * The column block is outermost and the depth block next, so a right panel is packed once for every strip of
 * the destination it serves rather than once per row block. The row block is innermost, which is what makes
 * the live piece of the destination one block rather than a whole column strip.
 *
 * One traversal, walked by the execution below and by the route that describes it. Writing the schedule out
 * twice is how a route comes to name blocks the call never cut, so there is one of it.
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
internal fun productBlockRows(kernels: DenseProductKernels, m: Int): Int =
    blockExtent(PRODUCT_BLOCK_ROWS, kernels.tileRows, m)

/** The destination columns one block covers, rounded to whole tiles. */
internal fun productBlockColumns(kernels: DenseProductKernels, n: Int): Int =
    blockExtent(PRODUCT_BLOCK_COLUMNS, kernels.tileColumns, n)

/** The shared-dimension steps one block accumulates, which needs no rounding. */
internal fun productBlockDepth(k: Int): Int = if (PRODUCT_BLOCK_DEPTH < k) PRODUCT_BLOCK_DEPTH else k

/**
 * Whether a window of the destination lies wholly outside the selected triangle, so nothing schedules it.
 *
 * The extents are the destination's own, and the diagonal is the destination's: these routines are square,
 * and the window's position in it is what decides. A window below a lower triangle's diagonal is entirely
 * selected and one above it entirely discarded, which is what [insideTriangle] and this answer between them.
 */
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
 * The register tiles a block straddling the diagonal is cut into, skipping the ones it may not write.
 *
 * One walk, taken by the execution below and by the route that describes it. `whole` is true where every
 * entry of a tile is selected, which is the tile that reaches the destination directly; a tile that
 * straddles accumulates first and only its selected entries are merged.
 */
internal inline fun forEachSelectedTile(
    kernels: DenseProductKernels,
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
    kernels: DenseProductKernels,
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
    var rightBase = 0
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
                packRightPanel(
                    b, bOffset, ldb, transposeB, step, column, depth, columnCount, right, 0, tileColumns,
                )
                rightStride = depth * tileColumns
                rightBase = 0
                packedColumn = column
                packedStep = step
            }
            packLeftPanel(a, aOffset, lda, transposeA, row, step, rowCount, depth, left, 0, tileRows)
            val leftStride = depth * tileRows
            val leftBase = 0
            // Apply beta only in the first depth block; later blocks accumulate.
            val blockBeta = if (step == 0) beta else 1.0
            if (insideTriangle(row, rowCount, column, columnCount, selected)) {
                kernels.productBlock(
                    alpha, left, leftBase, leftStride, right, rightBase, rightStride,
                    rowCount, columnCount, depth, blockBeta, c, cOffset + row + column * ldc, ldc,
                )
            } else {
                selectedBlock(
                    kernels, alpha, left, leftBase, leftStride, right, rightBase, rightStride,
                    row, rowCount, column, columnCount, depth, blockBeta, c, cOffset, ldc, selected, edge,
                )
            }
        }
    }
}

/**
 * One block straddling the diagonal, written a register tile at a time so only selected entries reach it.
 *
 * The tiles are the backend's own, and the panels are the enclosing block's: a tile reads the group its row
 * or column falls in, which is why this needs no packing of its own. A tile wholly inside the triangle is
 * an ordinary block of the backend's tile shape and goes to the destination directly. One that straddles
 * accumulates into [edge] with no destination multiplier, and the merge below is what spends that
 * multiplier, once, on the entries the call is allowed to write. A tile wholly outside runs nothing.
 *
 * How many tiles straddle is the tile shape's own answer and is not one: a row of tiles crosses the diagonal
 * over as many of them as the tile's rows cover columns. What is bounded is what a straddling tile discards,
 * which is part of one tile rather than part of a cache block, and that is the whole reason the merge
 * happens here rather than over the enclosing block.
 */
private fun selectedBlock(
    kernels: DenseProductKernels,
    alpha: Double,
    leftPanel: DoubleArray,
    leftBase: Int,
    leftStride: Int,
    rightPanel: DoubleArray,
    rightBase: Int,
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
        val leftTile = leftBase + tileRow / tileRows * leftStride
        val rightTile = rightBase + tileColumn / kernels.tileColumns * rightStride
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
 * The product as Level 2 panel work, one destination column at a time, without packing both operands.
 *
 * Untransposed on the left, panels accumulate in a bounded workspace column before destination writeback.
 * The thin-GEMM comparison found this faster than direct destination updates; alpha placement stays internal.
 * Transposed panels apply both multipliers on write. Neither form promises identical backend rounding.
 *
 * The coefficient vector is a column of `op(B)`, adjacent when `B` is untransposed and [ldb] apart when it
 * is not. A strided one is scalar work for a reduction whatever the width, so where the reduction is long
 * enough to pay for it the column is gathered once into adjacent storage and the reduction runs vectorised
 * against that. [gathersCoefficients] is where that choice is made, and the route asks the same question.
 *
 * A selected triangle shortens each destination column to the rows the call may write, and the panels it
 * hands over are that much of the operand. No entry outside the selected triangle is read or written.
 */
internal fun directProduct(
    panels: DensePanelKernels,
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
    workspace: Workspace?,
) {
    val coefficientStride = if (transposeB) ldb else 1
    if (transposeA) {
        reducedProduct(
            panels, alpha, a, aOffset, lda, b, bOffset, ldb, transposeB, coefficientStride,
            beta, c, cOffset, ldc, m, n, k, selected, workspace,
        )
        return
    }
    val chunk = directColumnBlock(m)
    val group = directPanelGroup(panels, m, k, transposeA = false)
    workspace.borrowOptional(scratchCapacity(chunk)) { accumulated ->
        forEachDirectProductPanel(m, n, k, selected, false, group) { column, first, step, height, width ->
            if (step == 0) accumulated.fill(0.0, 0, height)
            val coefficients = bOffset + if (transposeB) column else column * ldb
            panels.columnUpdate(
                1.0, a, aOffset + first + step * lda, lda,
                b, coefficients + step * coefficientStride, coefficientStride,
                height, width, accumulated, 0, 1,
            )
            if (step + width == k) {
                writeScaledColumn(
                    alpha,
                    accumulated,
                    beta,
                    c,
                    cOffset + first + column * ldc,
                    height,
                )
            }
        }
    }
}

/** Bounds the accumulating column independently of the call order and backend panel grouping. */
internal fun directColumnBlock(m: Int): Int = if (m < PRODUCT_BLOCK_ROWS) m else PRODUCT_BLOCK_ROWS

/**
 * The panel the unpacked route's arithmetic is, and how many of its columns the backend recommends handing
 * over at a time.
 *
 * A transposed left operand reduces down stored columns and an untransposed one accumulates them into the
 * destination, which are different panels over different shapes. The execution and the route ask here, so a
 * route cannot report a grouping the call did not use.
 */
internal fun directPanelWork(transposeA: Boolean): PanelWork =
    if (transposeA) PanelWork.MultiDot else PanelWork.ColumnUpdate

/** The grouping for [directPanelWork], over the panel shape that route really cuts. */
internal fun directPanelGroup(panels: DensePanelKernels, m: Int, k: Int, transposeA: Boolean): Int = if (transposeA) {
    panels.executionGroup(PanelWork.MultiDot, k, m)
} else {
    panels.executionGroup(PanelWork.ColumnUpdate, directColumnBlock(m), k)
}

/** The transposed-left half of [directProduct], with or without the gathered coefficient column. */
private fun reducedProduct(
    panels: DensePanelKernels,
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
    workspace: Workspace?,
) {
    val group = directPanelGroup(panels, m, k, transposeA = true)
    val gathers = gathersCoefficients(panels, m, k, coefficientStride != 1)
    workspace.borrowOptional(if (gathers) scratchCapacity(k) else 0) { gathered ->
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
 * Whether the direct route gathers a strided coefficient column into adjacent storage before reducing.
 *
 * Three conditions, and all of them have to hold for the copy to be worth making. The column has to be
 * strided, because an adjacent one is already what a vector body wants. The gather has to change which body
 * the reduction reaches, which is the backend's own answer and not an assumption about it. And there have to
 * be enough destination rows to read the gathered column back several times, since the copy costs one pass
 * over it and each destination row saves one strided pass.
 *
 * [DIRECT_GATHER_MINIMUM_ROWS] is where the copy is clearly ahead rather than where it first is. The local
 * evidence has the reduction run both ways over the same operands: from eight destination rows the copy
 * leads by a wide margin at every depth long enough to have something to read back, and at four rows the two
 * swap places between captures. That is one machine, and it establishes no crossover for another host.
 */
internal fun gathersCoefficients(panels: DensePanelKernels, rows: Int, depth: Int, strided: Boolean): Boolean {
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
 * Round scratch lengths so neighboring windows reuse workspace loans. Schedule blocks bound the
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
 * Whole tiles of [preferred], never more than the extent needs and never less than one tile.
 *
 * A block extent is a multiple of the tile so that every block but the last starts on a group boundary,
 * so every panel starts on a group boundary. A configured block smaller than one tile would
 * leave a block with no extent and a traversal that never advances, so one tile is the floor.
 */
internal fun blockExtent(preferred: Int, tile: Int, extent: Int): Int {
    val whole = ((extent.toLong() + tile - 1) / tile * tile).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
    val block = preferred / tile * tile
    val atLeastOne = if (block < tile) tile else block
    return if (atLeastOne < whole) atLeastOne else whole
}

/** Direct arithmetic windows, shared with route inspection, including selected support and tails. */
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
