@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, T, X

package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.dense.borrowOptional
import com.eignex.koblas.dense.forEachPanel
import com.eignex.koblas.sparse.SparsePanelKernels

/* Shared triangular scheduling over reusable sparse column and RHS-panel leaves. */

/** Walks triangle columns in dependency order without allocating a descending progression. */
internal inline fun forEachTriangleColumn(order: Int, forward: Boolean, action: (column: Int) -> Unit) {
    var step = 0
    while (step < order) {
        action(if (forward) step else order - 1 - step)
        step++
    }
}

/** Runs [block] with the diagonal of [a] borrowed from [workspace], or null when [unitDiag] takes it as 1. */
internal inline fun withExplicitDiagonal(
    a: SparseMatrix,
    n: Int,
    unitDiag: Boolean,
    workspace: Workspace?,
    crossinline block: (DoubleArray?) -> Unit,
) {
    if (unitDiag) {
        block(null)
    } else {
        workspace.borrow(n) { diagonal ->
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
 * right-hand side's arithmetic reaches another's.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the triangle, the block, and the four BLAS triangle flags
internal fun triangularLeftCore(
    kernels: SparsePanelKernels,
    solve: Boolean,
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    diagonal: DoubleArray?,
    workspace: Workspace?,
) {
    val n = a.rows
    val bd = b.values
    val sides = b.cols
    if (n == 0 || sides == 0) return
    val forward = lower != (transpose == solve)
    // The gathering direction reduces finished rows into a pivot and the scattering one spreads a pivot,
    // which are the two panel shapes a backend answers about separately.
    val plan = planRightHandSides(
        kernels,
        n,
        sides,
        a.nnz,
        copiedPerSide = 2L * n,
        nativelyContiguous = false,
        reduction = transpose,
    )
    val width = rhsWidth(plan)
    val staged = rhsStaged(plan)
    // Twice the width, because a panel scatter records which right-hand sides are live beside them. A call
    // whose panels are all one right-hand side wide writes the pivot out itself and reaches no panel, so it
    // takes no loan rather than leaving a length in the workspace that nothing reads.
    workspace.borrowOptional(if (width > 1 && sides > 1) 2 * width else 0) { work ->
        workspace.borrowOptional(if (staged) width * n else 0) { panel ->
            forEachPanel(sides, width) { columnStart, actual ->
                val dense = if (staged) panel else bd
                val offset = if (staged) 0 else columnStart * n
                val rhsStride = if (staged) 1 else n
                val indexStride = if (staged) width else 1
                if (staged) stageRhsPanel(bd, columnStart * n, n, 1, n, actual, panel, width)
                forEachTriangleColumn(n, forward) { j ->
                    val dj = diagonal?.get(j) ?: 1.0
                    if (actual == 1) {
                        if (!transpose) {
                            kernels.triangularScatterSingle(
                                solve, j, lower, unitDiag, dj, a.rowIndices, a.values,
                                a.colPointers[j], a.colPointers[j + 1], dense, offset, indexStride,
                            )
                        } else {
                            kernels.triangularGatherSingle(
                                solve, j, lower, unitDiag, dj, a.rowIndices, a.values,
                                a.colPointers[j], a.colPointers[j + 1], dense, offset, indexStride,
                            )
                        }
                    } else if (!transpose) {
                        kernels.triangularPanelScatter(
                            solve, j, lower, unitDiag, dj, a.rowIndices, a.values,
                            a.colPointers[j], a.colPointers[j + 1],
                            dense, offset, rhsStride, indexStride, actual, work,
                        )
                    } else {
                        kernels.triangularPanelGather(
                            solve, j, lower, unitDiag, dj, a.rowIndices, a.values,
                            a.colPointers[j], a.colPointers[j + 1],
                            dense, offset, rhsStride, indexStride, actual, work,
                        )
                    }
                }
                if (staged) unstageRhsPanel(panel, width, bd, columnStart * n, n, 1, n, actual)
            }
        }
    }
}

/**
 * Right multiply over contiguous dense columns, which turns every sparse update into a Level 1 operation,
 * the same trade [trsmRightCore] makes. A row times op(T) is op(T)ᵀ times its column-shaped view, so this
 * walks the triangle exactly as [triangularLeftCore] does with the transpose flag flipped, only every scalar
 * lane of that algorithm is a whole column of [b] here instead of one right-hand side in a panel.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the triangle, the block, and the three BLAS triangle flags
internal fun trmmRightCore(
    kernels: SparsePanelKernels,
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    diagonal: DoubleArray?,
) {
    val rows = b.rows
    if (rows == 0) return
    val n = a.rows
    val gather = !transpose
    forEachTriangleColumn(n, lower == gather) { l ->
        kernels.triangularRightColumn(
            false, gather, l, lower, unitDiag, diagonal?.get(l) ?: 1.0,
            a.rowIndices, a.values, a.colPointers[l], a.colPointers[l + 1], b.values, rows,
        )
    }
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
    kernels: SparsePanelKernels,
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
                kernels.triangularAxpy(j, lower, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], xj, x)
            }
        } else {
            val scaled = if (unitDiag) x[j] else a[j, j] * x[j]
            x[j] = kernels.triangularReduce(
                j, lower, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], x, scaled, subtract = false,
            )
        }
    }
}

/** Right solve over contiguous dense columns, which turns every sparse update into a Level 1 operation. */
@OptIn(UnsafeKoblasApi::class)
internal fun trsmRightCore(
    kernels: SparsePanelKernels,
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    diagonal: DoubleArray?,
) {
    val rows = b.rows
    if (rows == 0) return
    val gather = !transpose
    forEachTriangleColumn(a.rows, lower != transpose) { j ->
        kernels.triangularRightColumn(
            true, gather, j, lower, diagonal == null, diagonal?.get(j) ?: 1.0,
            a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], b.values, rows,
        )
    }
}

/** `trsv` over the `n` entries of [x], with the triangle flags resolved once by the caller. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the three BLAS triangle flags
internal fun trsvCore(
    kernels: SparsePanelKernels,
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
            kernels.triangularAxpy(j, lower, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], -xj, x)
        } else {
            val s = kernels.triangularReduce(
                j, lower, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], x, x[j], subtract = true,
            )
            x[j] = if (unitDiag) s else s / a[j, j]
        }
    }
}
