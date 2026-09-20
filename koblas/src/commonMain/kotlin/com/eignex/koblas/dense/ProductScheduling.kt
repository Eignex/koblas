@file:Suppress("LongParameterList") // a product carries both operands, both transpose flags and three extents

package com.eignex.koblas.dense

import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow

/*
 * Shared scheduling for `C = alpha * op(A) * op(B) + beta * C`, over any product backend.
 *
 * Two routes, and which one a call takes is a question about its extents rather than about its engine.
 *
 * A product large enough to hide a copy is cut into cache blocks, each operand block is packed into the
 * groups the backend's tile reads, and [DenseProductKernels.productBlock] accumulates the destination window
 * from them. The packing is portable Kotlin here, one copy per block, and the tile arithmetic is the
 * backend's. Everything about the traversal is this file's: which blocks, in which order, and which of them
 * carries `beta`.
 *
 * A product too small or too thin for that runs where the operands already are, as Level 2 panel work down
 * the columns of the destination. It avoids packing both operands; a strided coefficient column may still
 * be gathered, and an accumulating column uses scratch. The arithmetic uses the same panel kernels as the
 * matrix-vector routines.
 *
 * `beta` reaches an output window exactly once either way. In the blocked route the first depth block carries
 * it and every later one accumulates; in the panel route it is spent in the same pass that spends `alpha`,
 * which is the write at the end of a destination column or the write a reduction already makes.
 *
 * Every operand arrives as a window: an array, the offset its logical origin sits at and a leading dimension.
 * That is what lets a structured algorithm above this file hand over a strip of a triangle or a block of its
 * own right-hand sides without copying it out first, and it is why the offsets are flat. A caller computes
 * one from the transpose it is passing, since `op(A)`'s origin is `row + column · lda` where the operand is
 * stored as it reads and `column + row · lda` where it is transposed.
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
 * [borrow] for a loan a call may not need, which is one of no entries.
 *
 * A product between two retained panels copies nothing, a rectangular one selects no triangle and a
 * triangular call whose backend declines to gather copies no right-hand sides, so each asks for scratch of
 * no length. Taking that from the workspace would leave a zero length behind in it, which counts against
 * what it retains for the shapes the caller is actually working on, so it is not taken at all.
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
 * The product route a window of this shape takes, executed.
 *
 * One question asked in one place: a product with enough arithmetic to hide a copy of both operands is
 * packed into the backend's tiles, and one without runs as panel work over the operands where they lie.
 * Every caller in this library goes through here, so a structured algorithm handing over a strip of its own
 * takes the route that strip's extents earn rather than the one the whole call would have earned.
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
            kernels, alpha, a, aOffset, lda, transposeA, null, b, bOffset, ldb, transposeB, null,
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
 * Whether a window of these extents is packed into the backend's tiles.
 *
 * The backend is asked about the window's real extents, a selected-triangle one included. Those extents are
 * what the schedule hands it: the same cache blocks over the same operands, with the blocks lying wholly in
 * the other triangle dropped. A backend may answer from the depth, from a minimum dimension or from a tail
 * rather than from a product of the three, and a caller that passed it a smaller shared dimension to stand
 * for the arithmetic a triangle discards would be answering one of those questions on its behalf.
 *
 * So a triangle-selected window takes the rectangle's own eligibility. Such a window finishes about half
 * the arithmetic per copied value that the rectangle does, so its own crossover sits somewhere above the
 * rectangle's and this rule packs a band of shapes a little sooner than a rule of its own would. The
 * calibration left that as it is, for the reason [packsProductByWork] records: the rectangle's own
 * threshold did not separate its wins from its losses on the measured host either, so a second threshold
 * fitted beside it would be fitted to the same data. Route and execution both ask this one function, so
 * however it is answered they agree.
 */
@Suppress("UNUSED_PARAMETER") // the selected triangle is part of the question even where the answer ignores it
internal fun packsWindow(kernels: DenseProductKernels, m: Int, n: Int, k: Int, selected: OutputTriangle): Boolean =
    kernels.packsProduct(m, n, k)

/**
 * The blocked packed product, taking either operand already packed.
 *
 * A retained panel is used where it lies: its groups are as deep as the shared dimension it was packed over,
 * and [PackedLayout.groupStride] is what lets a block read a slice of one without repacking. A panel packed
 * for this call is as deep as its own block instead, and the same kernel reads both.
 *
 * Only the operands that are not already packed take scratch, so a product between two retained panels
 * borrows nothing at all. A selected triangle borrows one tile besides, which is where a block straddling
 * the diagonal accumulates before its selected entries are merged.
 */
internal fun blockedProduct(
    kernels: DenseProductKernels,
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    transposeA: Boolean,
    retainedA: PackedMatrix?,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    transposeB: Boolean,
    retainedB: PackedMatrix?,
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
    val leftSize = if (retainedA == null) scratchCapacity(blockRows * blockDepth) else 0
    val rightSize = if (retainedB == null) scratchCapacity(blockColumns * blockDepth) else 0
    val edgeSize = if (selected == OutputTriangle.Full) 0 else kernels.tileRows * kernels.tileColumns
    workspace.borrowOptional(leftSize) { left ->
        workspace.borrowOptional(rightSize) { right ->
            workspace.borrowOptional(edgeSize) { edge ->
                blockedProductCore(
                    kernels, alpha, a, aOffset, lda, transposeA, retainedA, left,
                    b, bOffset, ldb, transposeB, retainedB, right,
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

/** The block traversal itself, over panels that are either retained or packed into [left] and [right]. */
@Suppress("CyclomaticComplexMethod") // the traversal, its lazy packing and the three positions of a block
private fun blockedProductCore(
    kernels: DenseProductKernels,
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    transposeA: Boolean,
    retainedA: PackedMatrix?,
    left: DoubleArray,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    transposeB: Boolean,
    retainedB: PackedMatrix?,
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
    val leftPanel = retainedA?.values ?: left
    val rightPanel = retainedB?.values ?: right
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
        // A block in the triangle the call does not write is not scheduled at all, so neither operand is
        // packed for it and the destination beneath it is never touched.
        if (!outsideTriangle(row, rowCount, column, columnCount, selected)) {
            // The right panel belongs to the column and depth block, and the row loop is inside both, so it
            // is packed when either of them moves and read where it lies for every row block after that.
            if (column != packedColumn || step != packedStep) {
                if (retainedB == null) {
                    packRightPanel(
                        b, bOffset, ldb, transposeB, step, column, depth, columnCount, right, 0, tileColumns,
                    )
                    rightStride = depth * tileColumns
                    rightBase = 0
                } else {
                    rightStride = retainedB.layout.groupStride
                    rightBase = column / tileColumns * rightStride + step * tileColumns
                }
                packedColumn = column
                packedStep = step
            }
            val leftStride: Int
            val leftBase: Int
            if (retainedA == null) {
                packLeftPanel(a, aOffset, lda, transposeA, row, step, rowCount, depth, left, 0, tileRows)
                leftStride = depth * tileRows
                leftBase = 0
            } else {
                leftStride = retainedA.layout.groupStride
                leftBase = row / tileRows * leftStride + step * tileRows
            }
            // Every later depth block adds to what the first one left, so beta reaches an output window
            // once however the shared dimension was cut.
            val blockBeta = if (step == 0) beta else 1.0
            if (insideTriangle(row, rowCount, column, columnCount, selected)) {
                kernels.productBlock(
                    alpha, leftPanel, leftBase, leftStride, rightPanel, rightBase, rightStride,
                    rowCount, columnCount, depth, blockBeta, c, cOffset + row + column * ldc, ldc,
                )
            } else {
                selectedBlock(
                    kernels, alpha, leftPanel, leftBase, leftStride, rightPanel, rightBase, rightStride,
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
 * How many tiles straddle is the tile's shape's answer and is not one: a row of tiles crosses the diagonal
 * over as many of them as the tile's rows cover columns, so a tile of eight rows by four columns straddles
 * twice along each row and a tile whose sides were chosen independently may straddle more. What is bounded
 * is what a straddling tile discards, which is part of one tile rather than part of a cache block, and that
 * is the whole reason the merge happens here rather than over the enclosing block.
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
                val leftTile = leftBase + tileRow / tileRows * leftStride
                val rightTile = rightBase + tileColumn / tileColumns * rightStride
                val target = cOffset + at + from * ldc
                if (insideTriangle(at, rows, from, columns, selected)) {
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
            tileRow += tileRows
        }
        tileColumn += tileColumns
    }
}

/**
 * The selected entries of an accumulated tile, added into the destination with its multiplier spent once.
 *
 * A zero multiplier overwrites without reading, which is the destination's contract and the reason the
 * merge cannot simply add: a NaN standing in an output that the call is about to overwrite would survive.
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
 * Untransposed on the left, a destination column is the columns of `op(A)` accumulated with the coefficients
 * standing in that column of `op(B)`, which is [DensePanelKernels.columnUpdate]. The accumulation happens in
 * a borrowed column and the multipliers are spent once on the way out, so that `alpha` multiplies a sum of
 * products rather than each coefficient: the packed route multiplies a block sum, and a public call must not
 * change what `alpha` is applied to because its extents put it on one route or the other.
 *
 * Transposed on the left, a destination entry is one stored column of `A` reduced against that column of
 * `op(B)`, which is [DensePanelKernels.multiDot] with both multipliers folded into the write it already
 * makes. That is already `alpha` times a finished sum, so it needs no column of its own.
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
    val group = panels.executionGroup(PanelWork.ColumnUpdate, chunk, k)
    workspace.borrowOptional(scratchCapacity(chunk)) { accumulated ->
        for (j in 0 until n) {
            val start = selectedRow(j, m, selected)
            val rows = selectedRows(j, m, selected)
            var offset = 0
            while (offset < rows) {
                val height = if (chunk < rows - offset) chunk else rows - offset
                accumulated.fill(0.0, 0, height)
                val coefficients = bOffset + if (transposeB) j else j * ldb
                forEachPanel(k, group) { step, width ->
                    panels.columnUpdate(
                        1.0, a, aOffset + start + offset + step * lda, lda,
                        b, coefficients + step * coefficientStride, coefficientStride,
                        height, width, accumulated, 0, 1,
                    )
                }
                writeScaledColumn(alpha, accumulated, beta, c, cOffset + start + offset + j * ldc, height)
                offset += height
            }
        }
    }
}

/**
 * Destination rows the unpacked route accumulates at a time, which is what bounds the scratch it borrows.
 *
 * A destination column is accumulated in a borrowed buffer before both multipliers are spent on it, and
 * borrowing one as long as the column would make the buffer's length a function of the call's order. The
 * schedules above cut windows whose extents shrink, so that function takes a new value at every block, and
 * a workspace lending by exact length would hand back a buffer nothing asks for again. Cutting the column
 * instead puts a ceiling on the length independent of the order: every chunk but the last is exactly this
 * many rows, and the arithmetic is unchanged because the rows of a destination column are independent of
 * each other and the multipliers are still spent once on each.
 *
 * The same number as a cache block's rows, because it is the same question: how much of a destination to
 * keep live while the shared dimension is walked.
 */
internal fun directColumnBlock(m: Int): Int = if (m < PRODUCT_BLOCK_ROWS) m else PRODUCT_BLOCK_ROWS

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
    val group = panels.executionGroup(PanelWork.MultiDot, k, m)
    if (!gathersCoefficients(panels, m, k, coefficientStride != 1)) {
        for (j in 0 until n) {
            val first = selectedRow(j, m, selected)
            val rows = selectedRows(j, m, selected)
            val coefficients = bOffset + if (transposeB) j else j * ldb
            forEachPanel(rows, group) { start, width ->
                panels.multiDot(
                    alpha, a, aOffset + (first + start) * lda, lda, b, coefficients, coefficientStride,
                    k, width, beta, c, cOffset + first + start + j * ldc, 1,
                )
            }
        }
        return
    }
    workspace.borrowOptional(scratchCapacity(k)) { gathered ->
        for (j in 0 until n) {
            val first = selectedRow(j, m, selected)
            val rows = selectedRows(j, m, selected)
            if (rows <= 0) continue
            val coefficients = bOffset + if (transposeB) j else j * ldb
            for (p in 0 until k) gathered[p] = b[coefficients + p * coefficientStride]
            forEachPanel(rows, group) { start, width ->
                panels.multiDot(
                    alpha, a, aOffset + (first + start) * lda, lda, gathered, 0, 1, k, width,
                    beta, c, cOffset + first + start + j * ldc, 1,
                )
            }
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
 * the reduction reaches, which is the backend's own answer and not an assumption about it: a backend with
 * one body for both gains nothing. And there have to be enough destination rows to read the gathered column
 * back several times, since the copy costs one pass over it and each destination row saves one strided pass.
 *
 * The minimum below is where the copy is clearly ahead rather than where it first is. The stage evidence has
 * the reduction run both ways over the same operands: from eight destination rows the copy leads by a wide
 * margin at every depth long enough to have something to read back, and at four rows the two swap places
 * between captures. Where the answer is that unclear the route that copies nothing is the one to take. One
 * machine, and a crossover across machines is a later stage's.
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
 * `c = alpha · accumulated + beta · c` over one destination column.
 *
 * A zero [beta] overwrites without reading what is there, which is the destination's contract and not an
 * arithmetic shortcut: a NaN standing in the output would survive a multiply by zero.
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
 * The buffer length a scratch request of [size] is made at, which is the next power of two above it.
 *
 * A workspace lends by exact length and retains a bounded number of lengths, so scratch asked for at the
 * exact extent of a window is reused only where the windows repeat. A structured schedule's do not: a
 * triangular solve of order one thousand over a single right-hand side accumulates a destination column for
 * each of its sixteen diagonal blocks, and those columns shrink with the order. Every one of them would be
 * a fresh allocation on every warmed call, and a probe over a wide call or a short one would not see it.
 *
 * Rounding up collapses a run of nearby extents onto one buffer. It is not by itself a bound: an order
 * large enough spans more octaves than any retention holds, so what keeps the count bounded is that the
 * extents themselves are bounded, by [directColumnBlock] for an accumulating column and by the cache block
 * for a packed panel. Rounding is what absorbs what is left, which is the variation below those ceilings.
 * What it costs is a buffer up to twice the window it serves, which beside the operands a Level 3 call
 * already holds is small, and a caller reads only the entries it asked for. Above the largest power of two
 * an array length holds, the request passes through.
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
 * which is what lets a retained panel be read in slices. A configured block smaller than one tile would
 * leave a block with no extent and a traversal that never advances, so one tile is the floor.
 */
internal fun blockExtent(preferred: Int, tile: Int, extent: Int): Int {
    val whole = ((extent.toLong() + tile - 1) / tile * tile).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
    val block = preferred / tile * tile
    val atLeastOne = if (block < tile) tile else block
    return if (atLeastOne < whole) atLeastOne else whole
}
