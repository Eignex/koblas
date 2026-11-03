package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.sparse.*

/** The strictly off-diagonal part of one triangular column, added into a single right-hand side. */
@Suppress("LongParameterList") // the column slice and its triangle flag
internal fun sparseTriangularAxpy(
    column: Int,
    lower: Boolean,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    alpha: Double,
    destination: DoubleArray,
) {
    for (position in fromIndex until toIndex) {
        val row = rowIndices[position]
        if (if (lower) row > column else row < column) destination[row] += alpha * values[position]
    }
}

/** The strictly off-diagonal part of one triangular column, reduced against a single right-hand side. */
@Suppress("LongParameterList") // the column slice, its triangle flag, and the running sum
internal fun sparseTriangularReduce(
    column: Int,
    lower: Boolean,
    rowIndices: IntArray,
    values: DoubleArray,
    fromIndex: Int,
    toIndex: Int,
    source: DoubleArray,
    initial: Double,
    subtract: Boolean,
): Double {
    var sum = initial
    for (position in fromIndex until toIndex) {
        val row = rowIndices[position]
        if (if (lower) row > column else row < column) {
            val term = values[position] * source[row]
            sum = if (subtract) sum - term else sum + term
        }
    }
    return sum
}

/**
 * [sparseTriangularPanelScatter] with one right-hand side, written out for the reason [sparseGatherProductColumn]
 * gives.
 *
 * The liveness record the panel keeps is one boolean here, so it is the branch that returns rather than
 * a second half of a scratch array, and the one live right-hand side takes the same arithmetic the panel
 * takes when none of its own are masked.
 */
@Suppress("LongParameterList") // the column slice, its three triangle flags, and the dense window
internal fun sparseTriangularScatterSingle(
    solve: Boolean,
    column: Int,
    lower: Boolean,
    unitDiagonal: Boolean,
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
    dense[pivot] = when {
        solve -> coefficient
        unitDiagonal -> raw
        else -> diagonal * raw
    }
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
@Suppress("LongParameterList") // the column slice, its three triangle flags, and the dense window
internal fun sparseTriangularGatherSingle(
    solve: Boolean,
    column: Int,
    lower: Boolean,
    unitDiagonal: Boolean,
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
    var sum = if (solve || unitDiagonal) raw else diagonal * raw
    val start = strictRunStart(rowIndices, fromIndex, toIndex, column, lower)
    val end = strictRunEnd(rowIndices, fromIndex, toIndex, column, lower)
    for (position in start until end) {
        val term = values[position] * dense[offset + rowIndices[position] * indexStride]
        sum = if (solve) sum - term else sum + term
    }
    dense[pivot] = if (solve) sum / diagonal else sum
}

/**
 * Sparse `dtrmv`. The direction preserves each source before its destination is written, so no work buffer is
 * needed for the ordinary in-place case.
 *
 * The diagonal is probed as `A(j, j)`, a binary search per column. That is `trmv`'s to pay: `trmm` walks
 * several right-hand sides against one triangle and precomputes the diagonal for itself instead.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the three BLAS triangle flags
internal fun trmvCore(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    forEachTriangleColumn(a.rows, lower == transpose) { j ->
        if (!transpose) {
            val xj = x[j]
            if (xj != 0.0) {
                x[j] = if (unitDiag) xj else a[j, j] * xj
                sparseTriangularAxpy(
                    j,
                    lower,
                    a.rowIndices,
                    a.values,
                    a.colPointers[j],
                    a.colPointers[j + 1],
                    xj,
                    x,
                )
            }
        } else {
            val scaled = if (unitDiag) x[j] else a[j, j] * x[j]
            x[j] = sparseTriangularReduce(
                j, lower, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], x, scaled, subtract = false,
            )
        }
    }
}

/** `trsv` over the `n` entries of [x], with the triangle flags resolved once by the caller. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the three BLAS triangle flags
internal fun trsvCore(
    a: SparseMatrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
) {
    // Forward when a finished unknown feeds later columns, backward when it feeds earlier ones.
    forEachTriangleColumn(a.rows, lower != transpose) { j ->
        if (!transpose) {
            val raw = x[j]
            if (raw == 0.0) return@forEachTriangleColumn
            val xj = if (unitDiag) raw else raw / a[j, j]
            x[j] = xj
            sparseTriangularAxpy(
                j,
                lower,
                a.rowIndices,
                a.values,
                a.colPointers[j],
                a.colPointers[j + 1],
                -xj,
                x,
            )
        } else {
            val s = sparseTriangularReduce(
                j, lower, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], x, x[j], subtract = true,
            )
            x[j] = if (unitDiag) s else s / a[j, j]
        }
    }
}

/** Skip the right-hand sides whose zero pivots were masked by the triangular schedule. */
@Suppress("LongParameterList") // A CSC column slice and its masked destination panel.
internal fun sparseMaskedTriangularScatter(
    solve: Boolean,
    rowIndices: IntArray, values: DoubleArray, start: Int, end: Int,
    dense: DoubleArray, offset: Int, indexStride: Int, rhsStride: Int, width: Int,
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
