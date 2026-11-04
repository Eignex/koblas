package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.dense.scalarDivide
import com.eignex.koblas.internal.sparse.*
import com.eignex.koblas.sparse.*

/**
 * Dense right-hand sides to visit per walk of a sparse column, for a destination of [columns] of them.
 *
 * The sparse traversal is this file's, and the indices it reads are not something the dense panel
 * contract knows about. What it takes from that contract is the grouping, so that this width follows the
 * machine the same way the dense ones do instead of being a constant here. [contiguous] is part of the
 * question because a backend whose body over adjacent right-hand sides is a different body may want a
 * different number of them in it; what the scheduling adds on top is a ceiling of its own, for the
 * staging buffer a copy has to fit in.
 */
internal fun KoblasEngine.sparseRightHandSideGroup(
    rows: Int,
    columns: Int,
    contiguous: Boolean = false,
    reduction: Boolean = false,
): Int = panelProvider.executionGroup(sparsePanelWork(reduction), rows, columns, contiguous)

/**
 * Whether this backend does better over a panel of [width] adjacent right-hand sides than over a strided
 * one, which is what decides whether staging a dense block is worth considering at all.
 *
 * The backend's own answer rather than a comparison between two of its labels. A portable backend reads
 * one entry at a time either way and says no, so it never pays for a copy; a vector one says yes where
 * it has a whole lane block of them.
 */
internal fun KoblasEngine.sparsePrefersAdjacentSides(width: Int, entries: Int, reduction: Boolean = false): Boolean =
    panelProvider.prefersContiguous(sparsePanelWork(reduction), width, entries)

/** Reductions carry accumulators; scatters carry destination windows. */
private fun sparsePanelWork(reduction: Boolean): PanelWork =
    if (reduction) PanelWork.SparseRightHandSideReduction else PanelWork.SparseRightHandSides

/**
 * One CSC column of a symmetric product against a panel of right-hand sides.
 *
 * The stored triangle serves two halves of the product. Every selected entry scatters its column of the
 * dense block into the rows it stores, and every selected entry off the diagonal also gathers those rows
 * back into the column's own row, which is the half the triangle does not store. Both halves run over
 * the same contiguous run of the column's stored entries, because the rows ascend and the selected ones
 * are a prefix or a suffix of them with the diagonal at its inner end, so the two are one call of one
 * panel leaf and the diagonal is the position that leaf leaves out of the second half.
 *
 * One multiplier serves both halves, `alpha · value`, and the mirrored half accumulates into the
 * destination's pivot row as the walk reaches it rather than into a scratch that is added afterwards.
 * Neither half reads a gathered copy, so [width] right-hand sides of [b] serve both and the destination
 * is where every product lands.
 */
@Suppress("LongParameterList") // one CSC column slice against a dense panel, with both layouts
internal fun KoblasEngine.sparseSymmetricLeftColumn(
    alpha: Double,
    column: Int,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    b: DoubleArray,
    c: DoubleArray,
    offset: Int,
    rhsStride: Int,
    indexStride: Int,
    width: Int,
    lower: Boolean,
    cOffset: Int,
    cRhsStride: Int,
) {
    val start = selectedRunStart(rowIndices, fromIndex, toIndex, column, lower)
    val end = selectedRunEnd(rowIndices, fromIndex, toIndex, column, lower)
    if (start >= end) return
    val pivot = offset + column * indexStride
    // The diagonal scatters without mirroring and lies at the inner end of an ascending selected run.
    val diagonal = when {
        lower && rowIndices[start] == column -> 0
        !lower && rowIndices[end - 1] == column -> end - start - 1
        else -> -1
    }
    // Both halves read the same source layout; the mirrored half accumulates directly into the pivot row.
    panelProvider.indexedCoupledUpdate(
        alpha, c, b, offset, rhsStride, indexStride, rowIndices, values, start, end - start, width,
        pivot, c, cOffset + column * indexStride, diagonal, cOffset, cRhsStride, indexStride,
    )
}

/**
 * One output row of a transposed sparse product, reduced against a panel of dense right-hand sides.
 *
 * The reduction lands in [work], which is adjacent whatever the dense block's layout is, so this half of
 * the product vectorises as soon as the block's own right-hand sides are adjacent. [alpha] scales the
 * finished sum.
 */
@Suppress("LongParameterList") // the column slice, the dense window, and the output panel
internal fun KoblasEngine.sparseGatherProductPanel(
    alpha: Double,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    b: DoubleArray,
    offset: Int,
    rhsStride: Int,
    indexStride: Int,
    width: Int,
    c: DoubleArray,
    cOffset: Int,
    cRhsStride: Int,
    work: DoubleArray,
) {
    if (fromIndex == toIndex) return
    work.fill(0.0, 0, width)
    panelProvider.indexedColumnUpdate(
        1.0, b, offset, rhsStride, indexStride, rowIndices, values, fromIndex, toIndex - fromIndex,
        width, work, 0,
    )
    for (rhs in 0 until width) c[cOffset + rhs * cRhsStride] += alpha * work[rhs]
}

/**
 * One inner index of an untransposed sparse product, scattered across a panel of right-hand sides.
 *
 * The coefficients are gathered into [work] first, so the scattered half sees an adjacent source
 * whatever the dense operand's layout is; what decides whether this vectorises is the destination's own,
 * which is why a staged destination is what this direction of the product asks for.
 */
@Suppress("LongParameterList") // the column slice, the dense window, and the output panel
internal fun KoblasEngine.sparseScatterProductPanel(
    alpha: Double,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    b: DoubleArray,
    bOffset: Int,
    bRhsStride: Int,
    width: Int,
    c: DoubleArray,
    offset: Int,
    rhsStride: Int,
    indexStride: Int,
    work: DoubleArray,
) {
    for (rhs in 0 until width) work[rhs] = alpha * b[bOffset + rhs * bRhsStride]
    panelProvider.indexedRankUpdate(
        1.0, c, offset, rhsStride, indexStride, rowIndices, values, fromIndex, toIndex - fromIndex,
        width, work, 0,
    )
}

/**
 * One triangular column against a panel of right-hand sides, in the direction that finishes the pivot
 * first and then updates the rows below it.
 *
 * A right-hand side that is exactly zero is skipped, which is [KoblasEngine.trsv]'s rule carried to several
 * columns at once: it is what keeps a zero right-hand side from forming a product against a stored
 * infinity. Which ones are live is recorded in the second half of [work], so the update loop reads that
 * rather than the values again, and so no width is wider than the record can hold. [work] therefore needs
 * twice [width] entries here, where the other panels need [width].
 *
 * A group whose right-hand sides are all live is handed to the panel leaf instead, because with nothing
 * masked the two agree entry for entry: the skip only ever avoided a product with a zero. A group with
 * one dead right-hand side keeps the written-out loop, whatever the layout, so the rule survives on
 * every backend rather than becoming a property of the one that does not vectorise.
 */
@Suppress("LongParameterList") // the column slice, its two triangle flags, and the panel window
internal fun KoblasEngine.sparseTriangularPanelScatter(
    solve: Boolean,
    column: Int,
    lower: Boolean,
    diagonal: Double,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    dense: DoubleArray,
    offset: Int,
    rhsStride: Int,
    indexStride: Int,
    width: Int,
    work: DoubleArray,
) {
    var live = 0
    val pivot = offset + column * indexStride
    for (rhs in 0 until width) {
        val at = pivot + rhs * rhsStride
        val raw = dense[at]
        work[rhs] = when {
            !solve -> raw
            raw == 0.0 -> 0.0
            else -> raw / diagonal
        }
        work[width + rhs] = if (raw == 0.0) 0.0 else 1.0
        if (raw != 0.0) {
            live++
            dense[at] = if (solve) work[rhs] else diagonal * raw
        }
    }
    val start = strictRunStart(rowIndices, fromIndex, toIndex, column, lower)
    val end = strictRunEnd(rowIndices, fromIndex, toIndex, column, lower)
    if (start >= end) return
    if (live == width) {
        panelProvider.indexedRankUpdate(
            if (solve) -1.0 else 1.0, dense, offset, rhsStride, indexStride,
            rowIndices, values, start, end - start, width, work, 0,
        )
        return
    }
    sparseMaskedTriangularScatter(
        solve, rowIndices, values, start, end, dense, offset, indexStride, rhsStride, width, work,
    )
}

/**
 * One triangular column against a panel, in the direction that gathers the finished rows into the pivot.
 *
 * Nothing is masked here: this direction reads the rows it has already finished rather than spreading a
 * pivot into rows it has not, so every right-hand side contributes and the whole strictly triangular run
 * is one panel call. The accumulation lands in [work], which is adjacent whatever the block's layout is.
 */
@Suppress("LongParameterList") // the column slice, its two triangle flags, and the panel window
internal fun KoblasEngine.sparseTriangularPanelGather(
    solve: Boolean,
    column: Int,
    lower: Boolean,
    diagonal: Double,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    dense: DoubleArray,
    offset: Int,
    rhsStride: Int,
    indexStride: Int,
    width: Int,
    work: DoubleArray,
) {
    val pivot = offset + column * indexStride
    for (rhs in 0 until width) {
        val raw = dense[pivot + rhs * rhsStride]
        work[rhs] = if (solve) raw else diagonal * raw
    }
    val start = strictRunStart(rowIndices, fromIndex, toIndex, column, lower)
    val end = strictRunEnd(rowIndices, fromIndex, toIndex, column, lower)
    if (start < end) {
        panelProvider.indexedColumnUpdate(
            if (solve) -1.0 else 1.0, dense, offset, rhsStride, indexStride,
            rowIndices, values, start, end - start, width, work, 0,
        )
    }
    for (rhs in 0 until width) {
        val at = pivot + rhs * rhsStride
        dense[at] = if (solve) work[rhs] / diagonal else work[rhs]
    }
}

/**
 * One triangular column with the triangle on the right of the dense block, which makes every update a
 * whole contiguous dense column and therefore the selected Level 1 `axpy` and `scale`.
 *
 * The diagonal is divided by rather than multiplied by its reciprocal, so a right-hand solve rounds the
 * way [KoblasEngine.trsv] does over the same triangle.
 */
@Suppress("LongParameterList") // the column slice, its four flags, and the dense block
internal fun KoblasEngine.sparseTriangularRightColumn(
    solve: Boolean,
    gather: Boolean,
    column: Int,
    lower: Boolean,
    unitDiagonal: Boolean,
    diagonal: Double,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    dense: DoubleArray,
    rows: Int,
    origin: Int,
    leadingDimension: Int,
) {
    val columnOffset = origin + column * leadingDimension
    if (gather && !unitDiagonal) sparseApplyDiagonal(dense, columnOffset, rows, diagonal, solve)
    for (position in fromIndex until toIndex) {
        val row = rowIndices[position]
        val value = values[position]
        if (value != 0.0 && (if (lower) row > column else row < column)) {
            val alpha = if (solve) -value else value
            val rowOffset = origin + row * leadingDimension
            if (solve == gather) {
                this.axpy(dense, rowOffset, alpha, dense, columnOffset, rows)
            } else {
                this.axpy(dense, columnOffset, alpha, dense, rowOffset, rows)
            }
        }
    }
    if (!gather && !unitDiagonal) sparseApplyDiagonal(dense, columnOffset, rows, diagonal, solve)
}

private fun KoblasEngine.sparseApplyDiagonal(
    dense: DoubleArray,
    offset: Int,
    rows: Int,
    diagonal: Double,
    solve: Boolean,
) {
    if (solve) {
        scalarDivide(dense, offset, rows, diagonal)
    } else {
        this.scale(dense, offset, diagonal, rows)
    }
}

/**
 * Where the selected part of a CSC column starts, with the diagonal counted as selected.
 *
 * A column's rows ascend, so what a triangle selects from it is a prefix or a suffix rather than a scattered
 * subset. That is what lets a selected half be handed to a panel leaf as one run, and these four functions
 * are what decide which entries a panel is actually given.
 */
internal fun selectedRunStart(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) lowerBound(rowIndices, from, to, column) else from

/** Where the selected part of a column ends, the counterpart of [selectedRunStart]. */
internal fun selectedRunEnd(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) to else lowerBound(rowIndices, from, to, column + 1)

/**
 * Where the strictly triangular part of a column starts, which is the selected part without its diagonal.
 *
 * Without rather than minus: an ascending run has its diagonal at the inner end of the selected part if it
 * is stored at all, so dropping it is a bound one position further in and never a hole in the middle.
 */
internal fun strictRunStart(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) lowerBound(rowIndices, from, to, column + 1) else from

/** Where the strictly triangular part of a column ends, the counterpart of [strictRunStart]. */
internal fun strictRunEnd(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) to else lowerBound(rowIndices, from, to, column)

/**
 * The first position in the ascending run `[from, to)` whose row index is at least [row].
 */
internal fun lowerBound(rowIndices: IntArray, from: Int, to: Int, row: Int): Int {
    // Triangle-only storage puts each boundary at an endpoint, avoiding binary search.
    if (from >= to || rowIndices[from] >= row) return from
    if (rowIndices[to - 1] < row) return to
    var low = from + 1
    var high = to - 1
    while (low < high) {
        val middle = (low + high) ushr 1
        if (rowIndices[middle] < row) low = middle + 1 else high = middle
    }
    return low
}
