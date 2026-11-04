package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.*

/**
 * [sparseTriangularPanelScatter] with one right-hand side, written out for the reason [sparseGatherProductColumn]
 * gives.
 *
 * The liveness record the panel keeps is one boolean here, so it is the branch that returns rather than
 * a second half of a scratch array, and the one live right-hand side takes the same arithmetic the panel
 * takes when none of its own are masked.
 */
@Suppress("LongParameterList") // the column slice, its two triangle flags, and the dense window
internal fun sparseTriangularScatterSingle(
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
    indexStride: Int,
) {
    val pivot = offset + column * indexStride
    val raw = dense[pivot]
    // Skip a zero right-hand side to preserve its pivot and avoid spreading nonfinite coefficients.
    if (raw == 0.0) return
    val coefficient = if (solve) raw / diagonal else raw
    dense[pivot] = if (solve) coefficient else diagonal * raw
    val start = strictRunStart(rowIndices, fromIndex, toIndex, column, lower)
    val end = strictRunEnd(rowIndices, fromIndex, toIndex, column, lower)
    for (position in start until end) {
        val at = offset + rowIndices[position] * indexStride
        if (solve) {
            dense[at] -= values[position] * coefficient
        } else {
            dense[at] += values[position] * coefficient
        }
    }
}

/**
 * [sparseTriangularPanelGather] with one right-hand side, written out for the reason [sparseGatherProductColumn]
 * gives.
 *
 * The accumulator is a local rather than an entry of a scratch array, which is the whole of what a group
 * of one has to gain over the panel: the pivot is read once, summed into, and written back once.
 */
@Suppress("LongParameterList") // the column slice, its two triangle flags, and the dense window
internal fun sparseTriangularGatherSingle(
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
    indexStride: Int,
) {
    val pivot = offset + column * indexStride
    val raw = dense[pivot]
    var sum = if (solve) raw else diagonal * raw
    val start = strictRunStart(rowIndices, fromIndex, toIndex, column, lower)
    val end = strictRunEnd(rowIndices, fromIndex, toIndex, column, lower)
    for (position in start until end) {
        val term = values[position] * dense[offset + rowIndices[position] * indexStride]
        sum = if (solve) sum - term else sum + term
    }
    dense[pivot] = if (solve) sum / diagonal else sum
}

/**
 * Sparse `dtrsv` when [solve] and `dtrmv` otherwise, in place over [x]. The direction preserves each source
 * before its destination is written, so no work buffer is needed for the ordinary in-place case.
 *
 * The diagonal is probed as `A(j, j)`, a binary search per column, and a unit diagonal is taken as 1, which
 * multiplies and divides exactly. That is the vector routines' to pay: the matrix ones walk several
 * right-hand sides against one triangle and precompute the diagonal for themselves instead.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the triangle, the vector, and the four BLAS triangle flags
internal fun triangularVectorCore(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    solve: Boolean,
) {
    forEachTriangleColumn(a.rows, triangularForward(lower, transpose, solve)) { j ->
        // A zero scattered pivot is skipped before its diagonal is searched for.
        if (!transpose && x[j] == 0.0) return@forEachTriangleColumn
        val diagonal = if (unitDiag) 1.0 else a[j, j]
        val from = a.colPointers[j]
        val to = a.colPointers[j + 1]
        if (transpose) {
            sparseTriangularGatherSingle(solve, j, lower, diagonal, a.rowIndices, a.values, from, to, x, 0, 1)
        } else {
            sparseTriangularScatterSingle(solve, j, lower, diagonal, a.rowIndices, a.values, from, to, x, 0, 1)
        }
    }
}

/** Skip the right-hand sides whose zero pivots were masked by the triangular schedule. */
@Suppress("LongParameterList") // A CSC column slice and its masked destination panel.
internal fun sparseMaskedTriangularScatter(
    solve: Boolean,
    rowIndices: IntArray,
    values: DoubleArray,
    start: Int,
    end: Int,
    dense: DoubleArray,
    offset: Int,
    indexStride: Int,
    rhsStride: Int,
    width: Int,
    work: DoubleArray,
) {
    for (position in start until end) {
        val row = rowIndices[position]
        val value = values[position]
        val target = offset + row * indexStride
        for (rhs in 0 until width) {
            if (work[width + rhs] != 0.0) {
                val at = target + rhs * rhsStride
                if (solve) dense[at] -= value * work[rhs] else dense[at] += value * work[rhs]
            }
        }
    }
}
