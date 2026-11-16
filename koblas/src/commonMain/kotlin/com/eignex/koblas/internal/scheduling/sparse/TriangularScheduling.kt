@file:Suppress("FunctionParameterNaming", "LongParameterList", "VariableNaming") // Independent windows and extents.

package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.Scratch
import com.eignex.koblas.borrow
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.sparse.*
import com.eignex.koblas.internal.scheduling.forEachPanel
import com.eignex.koblas.internal.storage.borrowOptional
import com.eignex.util.arrays.lowerBoundInt

/* Shared triangular scheduling over reusable sparse column and RHS-panel leaves. */

/** Walks triangle columns in dependency order without allocating a descending progression. */
internal inline fun forEachTriangleColumn(order: Int, forward: Boolean, action: (column: Int) -> Unit) {
    var step = 0
    while (step < order) {
        action(if (forward) step else order - 1 - step)
        step++
    }
}

/**
 * Which direction a triangular dependence runs.
 */
internal fun triangularForward(lower: Boolean, transpose: Boolean, solve: Boolean): Boolean =
    lower != (transpose == solve)

/**
 * The plan a triangular routine takes over a block of right-hand sides.
 *
 * The block is both the source and the destination, so a staged copy is read in and written back, and the
 * direction that gathers finished rows into a pivot is the shape the backend is asked about.
 */
internal fun triangularRhsPlan(
    kernels: KernelProvider,
    order: Int,
    sides: Int,
    entries: Int,
    transpose: Boolean,
): RhsPlan = planRightHandSides(
    kernels,
    order,
    sides,
    entries,
    copiedPerSide = 2L * order,
    nativelyContiguous = false,
    reduction = transpose,
)

/** Borrows the diagonal from [aValues], or supplies null when [unitDiag] takes it as 1. */
internal inline fun withExplicitDiagonal(
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    n: Int,
    unitDiag: Boolean,
    arrays: Scratch?,
    crossinline block: (DoubleArray?) -> Unit,
) {
    if (unitDiag) {
        block(null)
    } else {
        arrays.borrow(n) { diagonal ->
            for (j in 0 until n) diagonal[j] = sparseDiagonal(aPointers, aIndices, aValues, j)
            block(diagonal)
        }
    }
}

/**
 * Sparse triangular solve or multiply shares a CSC traversal, differing at each pivot. Columns follow
 * triangle dependencies; right-hand sides are independent and may vectorise or be staged as a whole panel.
 * A null [diagonal] represents a unit diagonal.
 */
@Suppress("LongParameterList") // the triangle, the block, and the three BLAS triangle flags
internal fun triangularLeftCore(
    kernels: KernelProvider,
    solve: Boolean,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aRows: Int,
    aColumns: Int,
    bValues: DoubleArray,
    bOffset: Int,
    bColumns: Int,
    bLeadingDimension: Int,
    lower: Boolean,
    transpose: Boolean,
    diagonal: DoubleArray?,
    arrays: Scratch?,
) {
    val n = aRows
    val sides = bColumns
    if (n == 0 || sides == 0) return
    val forward = triangularForward(lower, transpose, solve)
    // Gathering reduces finished rows into a pivot; scattering spreads a pivot. Backends tune them separately.
    val plan = triangularRhsPlan(kernels, n, sides, aValues.size, transpose)
    val width = plan.width
    val ld = bLeadingDimension
    // Scatter scratch holds values and live flags. Single-right-hand-side groups need no panel loan.
    arrays.borrowOptional(if (width > 1 && sides > 1) 2 * width else 0) { work ->
        arrays.borrowOptional(if (plan.staged) width * n else 0) { panel ->
            forEachPanel(sides, width) { columnStart, actual ->
                withRhsPanel(
                    bValues, bOffset + columnStart * ld, ld, 1, n, actual, plan.staged, panel, width,
                    writeBack = true,
                ) { dense, offset, rhsStride, indexStride ->
                    forEachSparseColumn(aPointers, aColumns, forward) { j, from, to ->
                        val dj = diagonal?.get(j) ?: 1.0
                        if (actual == 1) {
                            if (!transpose) {
                                sparseTriangularScatterSingle(
                                    solve, j, lower, dj, aIndices, aValues, from, to, dense, offset, indexStride,
                                )
                            } else {
                                sparseTriangularGatherSingle(
                                    solve, j, lower, dj, aIndices, aValues, from, to, dense, offset, indexStride,
                                )
                            }
                        } else if (!transpose) {
                            kernels.sparseTriangularPanelScatter(
                                solve, j, lower, dj, aIndices, aValues, from, to,
                                dense, offset, rhsStride, indexStride, actual, work,
                            )
                        } else {
                            kernels.sparseTriangularPanelGather(
                                solve, j, lower, dj, aIndices, aValues, from, to,
                                dense, offset, rhsStride, indexStride, actual, work,
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Right-side triangular solve or multiply uses `op(T)ᵀ` per row, flipping [triangularLeftCore]'s traversal.
 * Each scalar update becomes a contiguous column Level 1 operation; [gather] follows the flipped direction.
 * A null [diagonal] represents a unit diagonal.
 */
@Suppress("LongParameterList") // the triangle, the block, and the three BLAS triangle flags
internal fun triangularRightCore(
    kernels: KernelProvider,
    solve: Boolean,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aRows: Int,
    bValues: DoubleArray,
    bOffset: Int,
    bRows: Int,
    bLeadingDimension: Int,
    lower: Boolean,
    gather: Boolean,
    diagonal: DoubleArray?,
) {
    val rows = bRows
    if (rows == 0) return
    forEachTriangleColumn(aRows, lower == gather) { j ->
        kernels.sparseTriangularRightColumn(
            solve, gather, j, lower, diagonal == null, diagonal?.get(j) ?: 1.0,
            aIndices, aValues, aPointers[j], aPointers[j + 1], bValues, rows, bOffset,
            bLeadingDimension,
        )
    }
}

/** Pure column/support enumeration, without coefficient reads. */
internal inline fun forEachSparseColumn(
    aPointers: IntArray,
    aColumns: Int,
    forward: Boolean = true,
    action: (column: Int, from: Int, to: Int) -> Unit,
) {
    forEachTriangleColumn(aColumns, forward) { column ->
        action(column, aPointers[column], aPointers[column + 1])
    }
}

/**
 * In-place sparse `dtrsv` or `dtrmv` preserves sources by traversal direction, requiring no work buffer.
 * Each column binary-searches its diagonal, treating unit diagonals as 1. Matrix routines instead precompute
 * diagonals once for reuse across right-hand sides.
 */
@Suppress("LongParameterList") // the triangle, the vector, and the four BLAS triangle flags
internal fun triangularVectorCore(
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aRows: Int,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    solve: Boolean,
) {
    forEachTriangleColumn(aRows, triangularForward(lower, transpose, solve)) { j ->
        // A zero scattered pivot is skipped before its diagonal is searched for.
        if (!transpose && x[j] == 0.0) return@forEachTriangleColumn
        val diagonal = if (unitDiag) 1.0 else sparseDiagonal(aPointers, aIndices, aValues, j)
        val from = aPointers[j]
        val to = aPointers[j + 1]
        if (transpose) {
            sparseTriangularGatherSingle(solve, j, lower, diagonal, aIndices, aValues, from, to, x, 0, 1)
        } else {
            sparseTriangularScatterSingle(solve, j, lower, diagonal, aIndices, aValues, from, to, x, 0, 1)
        }
    }
}

/** Reads a stored diagonal or the implicit zero without reconstructing a sparse container. */
private fun sparseDiagonal(pointers: IntArray, indices: IntArray, values: DoubleArray, column: Int): Double {
    val end = pointers[column + 1]
    val position = indices.lowerBoundInt(column, pointers[column], end)
    return if (position < end && indices[position] == column) values[position] else 0.0
}
