@file:Suppress("FunctionParameterNaming", "LongParameterList", "VariableNaming") // Independent windows and extents.

package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.scalar.sparse.*
import com.eignex.koblas.internal.operations.*
import com.eignex.koblas.internal.scheduling.forEachPanel
import com.eignex.koblas.internal.storage.borrowOptional

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
    kernels: KoblasEngine,
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

/** Runs [block] with the diagonal of [a] borrowed from reusable scratch, or null when [unitDiag] takes it as 1. */
internal inline fun withExplicitDiagonal(
    a: SparseMatrix,
    n: Int,
    unitDiag: Boolean,
    arrays: Scratch?,
    crossinline block: (DoubleArray?) -> Unit,
) {
    if (unitDiag) {
        block(null)
    } else {
        arrays.borrow(n) { diagonal ->
            for (j in 0 until n) diagonal[j] = a[j, j]
            block(diagonal)
        }
    }
}

/**
 * Sparse triangular multiply or solve over panels of right-hand sides, so a column's values and indices are
 * read once for several of them.
 *
 * One traversal for both, because they are the same walk with the same panels and differ in what each column
 * does at its pivot. The dependence between columns is the triangle's and stays in the order below; the
 * right-hand sides of a panel are independent of each other, which is what a panel leaf may vectorise.
 *
 * Where the block's right-hand sides are worth staging adjacent, the whole panel is solved or multiplied in
 * the staged copy and written back, which is correct for the same reason the panel exists: nothing in one
 * right-hand side's arithmetic reaches another's. A null [diagonal] is a unit one, taken as 1.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the triangle, the block, and the three BLAS triangle flags
internal fun triangularLeftCore(
    kernels: KoblasEngine,
    solve: Boolean,
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    diagonal: DoubleArray?,
    arrays: Scratch?,
) {
    val n = a.rows
    val sides = b.cols
    if (n == 0 || sides == 0) return
    val forward = triangularForward(lower, transpose, solve)
    // Gathering reduces finished rows into a pivot; scattering spreads a pivot. Backends tune them separately.
    val plan = triangularRhsPlan(kernels, n, sides, a.nnz, transpose)
    val width = plan.width
    val ld = b.leadingDimension
    // Scatter scratch holds values and live flags. Single-right-hand-side groups need no panel loan.
    arrays.borrowOptional(if (width > 1 && sides > 1) 2 * width else 0) { work ->
        arrays.borrowOptional(if (plan.staged) width * n else 0) { panel ->
            forEachPanel(sides, width) { columnStart, actual ->
                withRhsPanel(
                    b.values, b.offset + columnStart * ld, ld, 1, n, actual, plan.staged, panel, width,
                    writeBack = true,
                ) { dense, offset, rhsStride, indexStride ->
                    forEachSparseColumn(a, forward) { j, from, to ->
                        val dj = diagonal?.get(j) ?: 1.0
                        if (actual == 1) {
                            if (!transpose) {
                                sparseTriangularScatterSingle(
                                    solve, j, lower, dj, a.rowIndices, a.values, from, to, dense, offset, indexStride,
                                )
                            } else {
                                sparseTriangularGatherSingle(
                                    solve, j, lower, dj, a.rowIndices, a.values, from, to, dense, offset, indexStride,
                                )
                            }
                        } else if (!transpose) {
                            kernels.sparseTriangularPanelScatter(
                                solve, j, lower, dj, a.rowIndices, a.values, from, to,
                                dense, offset, rhsStride, indexStride, actual, work,
                            )
                        } else {
                            kernels.sparseTriangularPanelGather(
                                solve, j, lower, dj, a.rowIndices, a.values, from, to,
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
 * Triangular multiply or solve with the triangle on the right of contiguous dense columns, which turns every
 * sparse update into a Level 1 operation. A row times op(T) is op(T)ᵀ times its column-shaped view, so this
 * walks the triangle as [triangularLeftCore] does with the transpose flag flipped, only every scalar lane of
 * that algorithm is a whole column of [b] here instead of one right-hand side in a panel. [gather] is that
 * flipped direction's, and a null [diagonal] is a unit one.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the triangle, the block, and the three BLAS triangle flags
internal fun triangularRightCore(
    kernels: KoblasEngine,
    solve: Boolean,
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    gather: Boolean,
    diagonal: DoubleArray?,
) {
    val rows = b.rows
    if (rows == 0) return
    forEachTriangleColumn(a.rows, lower == gather) { j ->
        kernels.sparseTriangularRightColumn(
            solve, gather, j, lower, diagonal == null, diagonal?.get(j) ?: 1.0,
            a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], b.values, rows, b.offset,
            b.leadingDimension,
        )
    }
}

/** Sparse `trsm` when [solve] and `trmm` otherwise, from either side. */
internal fun KoblasEngine.sparseTriangularMatrix(
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    right: Boolean,
    alpha: Double,
    solve: Boolean,
    arrays: Scratch?,
) {
    requireTriangularMatrixOperands(a, b, right, if (solve) "trsm" else "trmm")
    if (alpha == 0.0) return scaleRegion(b, 0.0)
    val n = a.rows
    if (n == 0 || (if (right) b.rows else b.cols) == 0) return
    // Snapshot an aliased triangle before alpha scales the right-hand sides and changes its coefficients.
    staged(arrays, a, a.values === b.values) { triangle ->
        scaleRegion(b, alpha, vectors = this)
        withExplicitDiagonal(triangle, n, unitDiag, arrays) { diagonal ->
            if (right) {
                triangularRightCore(this, solve, triangle, b, lower, solve == transpose, diagonal)
            } else {
                triangularLeftCore(this, solve, triangle, b, lower, transpose, diagonal, arrays)
            }
        }
    }
}

/** Pure column/support enumeration, without coefficient reads. */
@OptIn(UnsafeKoblasApi::class)
internal inline fun forEachSparseColumn(
    a: SparseMatrix,
    forward: Boolean = true,
    action: (column: Int, from: Int, to: Int) -> Unit,
) {
    forEachTriangleColumn(a.cols, forward) { column ->
        action(column, a.colPointers[column], a.colPointers[column + 1])
    }
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
