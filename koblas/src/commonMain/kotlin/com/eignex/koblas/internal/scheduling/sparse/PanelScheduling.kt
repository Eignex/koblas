package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.panelProvider
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.dense.scalarDivide
import com.eignex.koblas.internal.kernels.scalar.sparse.sparseMaskedTriangularScatter
import com.eignex.koblas.internal.storage.sparse.*

/**
 * Backend-recommended grouping for the current right-hand-side layout, capped by scheduling's staging budget.
 * The traversal remains portable CSC; a grouping recommendation does not imply vectorised sparse indices.
 */
internal fun KoblasEngine.sparseRightHandSideGroup(
    rows: Int,
    columns: Int,
    contiguous: Boolean = false,
    reduction: Boolean = false,
): Int = panelProvider.executionGroup(sparsePanelWork(reduction), rows, columns, contiguous)

/**
 * Whether adjacent panels of [width] sides outperform strided ones. This is an explicit backend preference;
 * portable kernels decline staging because their arithmetic gains nothing from adjacency.
 */
internal fun KoblasEngine.sparsePrefersAdjacentSides(width: Int, entries: Int, reduction: Boolean = false): Boolean =
    panelProvider.prefersContiguous(sparsePanelWork(reduction), width, entries)

/** Reductions carry accumulators; scatters carry destination windows. */
private fun sparsePanelWork(reduction: Boolean): PanelWork =
    if (reduction) PanelWork.SparseRightHandSideReduction else PanelWork.SparseRightHandSides

/**
 * One selected CSC run serves both symmetric halves: scatter every entry and gather its mirror, excluding
 * the diagonal from the second half. Ascending rows make the selected triangle a contiguous prefix or suffix.
 * Both halves share `alpha · value` and accumulate directly into the destination without gathered scratch.
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
 * Reduce a transposed sparse column against dense right-hand sides into adjacent [work].
 * Vectorisation depends on the dense source layout; [alpha] scales the finished sum.
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
 * Scatter an untransposed sparse column using coefficients gathered into adjacent [work].
 * Vectorisation depends on the destination layout, so this direction may stage the destination.
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
 * Finish a triangular pivot, then scatter it. Exactly-zero right-hand sides are skipped, preserving TRSV's
 * no-read rule against stored infinities. The second half of [work] records live sides, requiring twice [width].
 * Only wholly live groups use the panel leaf; partially masked groups retain the explicit loop on every backend.
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
 * Gather finished triangular rows into the pivot through adjacent [work]. Unlike scatter, this direction
 * masks nothing: every stored strictly triangular entry participates.
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
 * Right-side triangular updates use contiguous column Level 1 operations. Divide by the diagonal directly;
 * reciprocal multiplication would change the solve's division semantics.
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
                kernels.axpy(dense, rowOffset, alpha, dense, columnOffset, rows)
            } else {
                kernels.axpy(dense, columnOffset, alpha, dense, rowOffset, rows)
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
        kernels.scale(dense, offset, diagonal, rows)
    }
}
