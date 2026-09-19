@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, T, X

package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.sparse.SparsePanelKernels

/* Shared triangular scheduling over reusable sparse column and RHS-panel leaves. */

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

/** Sparse triangular multiply over RHS panels, so values and indices are read once for several dense columns. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the triangle, the block, and the three BLAS triangle flags
internal fun trmmLeftCore(
    kernels: SparsePanelKernels,
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    diagonal: DoubleArray?,
    work: DoubleArray,
    group: Int,
) {
    val n = a.rows
    val bd = b.values
    val order = if (lower != transpose) n - 1 downTo 0 else 0 until n
    forEachRhsPanel(b.cols, group) { columnStart, width ->
        for (j in order) {
            val dj = diagonal?.get(j) ?: 1.0
            if (!transpose) {
                kernels.triangularPanelScatter(
                    false, j, lower, unitDiag, dj, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1],
                    bd, n, columnStart, width, work,
                )
            } else {
                kernels.triangularPanelGather(
                    false, j, lower, unitDiag, dj, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1],
                    bd, n, columnStart, width, work,
                )
            }
        }
    }
}

/**
 * Right multiply over contiguous dense columns, which turns every sparse update into a Level 1 operation,
 * the same trade [trsmRightCore] makes. A row times op(T) is op(T)ᵀ times its column-shaped view, so this
 * walks the triangle exactly as [trmmLeftCore] does with the transpose flag flipped, only every scalar lane
 * of that algorithm is a whole column of [b] here instead of one right-hand side in a panel.
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
    val order = if (lower != gather) n - 1 downTo 0 else 0 until n
    for (l in order) {
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
    val n = a.rows
    if (!transpose) {
        val order = if (lower) n - 1 downTo 0 else 0 until n
        for (j in order) {
            val xj = x[j]
            if (xj != 0.0) {
                x[j] = if (unitDiag) xj else a[j, j] * xj
                kernels.triangularAxpy(j, lower, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], xj, x)
            }
        }
    } else {
        val order = if (lower) 0 until n else n - 1 downTo 0
        for (j in order) {
            val scaled = if (unitDiag) x[j] else a[j, j] * x[j]
            x[j] = kernels.triangularReduce(
                j, lower, a.rowIndices, a.values, a.colPointers[j], a.colPointers[j + 1], x, scaled, subtract = false,
            )
        }
    }
}

/** Sparse substitution over RHS panels, so values and indices are read once for several dense columns. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the triangle, the block, and the two remaining triangle flags
internal fun trsmLeftCore(
    kernels: SparsePanelKernels,
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    diagonal: DoubleArray?,
    work: DoubleArray,
    group: Int,
) {
    val n = a.rows
    val bd = b.values
    val order = if (lower != transpose) 0 until n else n - 1 downTo 0
    forEachRhsPanel(b.cols, group) { columnStart, width ->
        for (j in order) {
            val divisor = diagonal?.get(j) ?: 1.0
            if (!transpose) {
                kernels.triangularPanelScatter(
                    true, j, lower, diagonal == null, divisor, a.rowIndices, a.values,
                    a.colPointers[j], a.colPointers[j + 1], bd, n, columnStart, width, work,
                )
            } else {
                kernels.triangularPanelGather(
                    true, j, lower, diagonal == null, divisor, a.rowIndices, a.values,
                    a.colPointers[j], a.colPointers[j + 1], bd, n, columnStart, width, work,
                )
            }
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
    val order = if (lower != transpose) 0 until a.rows else a.rows - 1 downTo 0
    for (j in order) {
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
    val n = a.rows
    // Forward when a finished unknown feeds later columns, backward when it feeds earlier ones.
    val order = if (lower != transpose) 0 until n else n - 1 downTo 0
    for (j in order) {
        if (!transpose) {
            val raw = x[j]
            if (raw == 0.0) continue
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
