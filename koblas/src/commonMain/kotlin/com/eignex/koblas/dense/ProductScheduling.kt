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
 * The blocked packed product, taking either operand already packed.
 *
 * A retained panel is used where it lies: its groups are as deep as the shared dimension it was packed over,
 * and [PackedLayout.groupStride] is what lets a block read a slice of one without repacking. A panel packed
 * for this call is as deep as its own block instead, and the same kernel reads both.
 *
 * Only the operands that are not already packed take scratch, so a product between two retained panels
 * borrows nothing at all.
 */
internal fun blockedProduct(
    kernels: DenseProductKernels,
    alpha: Double,
    a: DoubleArray,
    lda: Int,
    transposeA: Boolean,
    retainedA: PackedMatrix?,
    b: DoubleArray,
    ldb: Int,
    transposeB: Boolean,
    retainedB: PackedMatrix?,
    beta: Double,
    c: DoubleArray,
    ldc: Int,
    m: Int,
    n: Int,
    k: Int,
    workspace: Workspace?,
) {
    val blockRows = productBlockRows(kernels, m)
    val blockColumns = productBlockColumns(kernels, n)
    val blockDepth = productBlockDepth(k)
    val leftSize = if (retainedA == null) blockRows * blockDepth else 0
    val rightSize = if (retainedB == null) blockColumns * blockDepth else 0
    if (retainedA != null && retainedB != null) {
        blockedProductCore(
            kernels, alpha, a, lda, transposeA, retainedA, NO_PANEL, b, ldb, transposeB, retainedB, NO_PANEL,
            beta, c, ldc, m, n, k, blockRows, blockColumns, blockDepth,
        )
        return
    }
    workspace.borrow(leftSize) { left ->
        workspace.borrow(rightSize) { right ->
            blockedProductCore(
                kernels, alpha, a, lda, transposeA, retainedA, left, b, ldb, transposeB, retainedB, right,
                beta, c, ldc, m, n, k, blockRows, blockColumns, blockDepth,
            )
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

/** The block traversal itself, over panels that are either retained or packed into [left] and [right]. */
private fun blockedProductCore(
    kernels: DenseProductKernels,
    alpha: Double,
    a: DoubleArray,
    lda: Int,
    transposeA: Boolean,
    retainedA: PackedMatrix?,
    left: DoubleArray,
    b: DoubleArray,
    ldb: Int,
    transposeB: Boolean,
    retainedB: PackedMatrix?,
    right: DoubleArray,
    beta: Double,
    c: DoubleArray,
    ldc: Int,
    m: Int,
    n: Int,
    k: Int,
    blockRows: Int,
    blockColumns: Int,
    blockDepth: Int,
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
        // The right panel belongs to the column and depth block, and the row loop is inside both, so it is
        // packed when either of them moves and read where it lies for every row block after that.
        if (column != packedColumn || step != packedStep) {
            if (retainedB == null) {
                packRightPanel(b, ldb, transposeB, step, column, depth, columnCount, right, 0, tileColumns)
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
            packLeftPanel(a, lda, transposeA, row, step, rowCount, depth, left, 0, tileRows)
            leftStride = depth * tileRows
            leftBase = 0
        } else {
            leftStride = retainedA.layout.groupStride
            leftBase = row / tileRows * leftStride + step * tileRows
        }
        kernels.productBlock(
            alpha, leftPanel, leftBase, leftStride, rightPanel, rightBase, rightStride,
            rowCount, columnCount, depth,
            // Every later depth block adds to what the first one left, so beta reaches an output window
            // once however the shared dimension was cut.
            if (step == 0) beta else 1.0,
            c, row + column * ldc, ldc,
        )
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
 */
internal fun directProduct(
    panels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    lda: Int,
    transposeA: Boolean,
    b: DoubleArray,
    ldb: Int,
    transposeB: Boolean,
    beta: Double,
    c: DoubleArray,
    ldc: Int,
    m: Int,
    n: Int,
    k: Int,
    workspace: Workspace?,
) {
    val coefficientStride = if (transposeB) ldb else 1
    if (transposeA) {
        reducedProduct(panels, alpha, a, lda, b, ldb, transposeB, coefficientStride, beta, c, ldc, m, n, k, workspace)
        return
    }
    val group = panels.executionGroup(PanelWork.ColumnUpdate, m, k)
    workspace.borrow(m) { accumulated ->
        for (j in 0 until n) {
            accumulated.fill(0.0, 0, m)
            val coefficients = if (transposeB) j else j * ldb
            forEachPanel(k, group) { start, width ->
                panels.columnUpdate(
                    1.0, a, start * lda, lda, b, coefficients + start * coefficientStride, coefficientStride,
                    m, width, accumulated, 0, 1,
                )
            }
            writeScaledColumn(alpha, accumulated, beta, c, j * ldc, m)
        }
    }
}

/** The transposed-left half of [directProduct], with or without the gathered coefficient column. */
private fun reducedProduct(
    panels: DensePanelKernels,
    alpha: Double,
    a: DoubleArray,
    lda: Int,
    b: DoubleArray,
    ldb: Int,
    transposeB: Boolean,
    coefficientStride: Int,
    beta: Double,
    c: DoubleArray,
    ldc: Int,
    m: Int,
    n: Int,
    k: Int,
    workspace: Workspace?,
) {
    val group = panels.executionGroup(PanelWork.MultiDot, k, m)
    if (!gathersCoefficients(panels, m, k, coefficientStride != 1)) {
        for (j in 0 until n) {
            val coefficients = if (transposeB) j else j * ldb
            forEachPanel(m, group) { start, width ->
                panels.multiDot(
                    alpha, a, start * lda, lda, b, coefficients, coefficientStride, k, width,
                    beta, c, start + j * ldc, 1,
                )
            }
        }
        return
    }
    workspace.borrow(k) { gathered ->
        for (j in 0 until n) {
            val coefficients = if (transposeB) j else j * ldb
            for (p in 0 until k) gathered[p] = b[coefficients + p * coefficientStride]
            forEachPanel(m, group) { start, width ->
                panels.multiDot(
                    alpha, a, start * lda, lda, gathered, 0, 1, k, width, beta, c, start + j * ldc, 1,
                )
            }
        }
    }
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
