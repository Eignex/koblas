@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, T, X

package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.sparse.PortableSparsePanelKernels
import com.eignex.koblas.sparse.REFERENCE_SPARSE_RHS_WIDTH

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
internal fun trmmLeftCore(
    kernels: PortableSparsePanelKernels,
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    unitDiag: Boolean,
    diagonal: DoubleArray?,
) {
    val n = a.rows
    val bd = b.data
    val work = DoubleArray(REFERENCE_SPARSE_RHS_WIDTH)
    val order = if (lower != transpose) n - 1 downTo 0 else 0 until n
    forEachRhsPanel(b.cols) { columnStart, width ->
        for (j in order) {
            val dj = diagonal?.get(j) ?: 1.0
            if (!transpose) {
                kernels.triangularPanelScatter(
                    false, j, lower, unitDiag, dj, a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1],
                    bd, n, columnStart, width, work,
                )
            } else {
                kernels.triangularPanelGather(
                    false, j, lower, unitDiag, dj, a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1],
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
internal fun trmmRightCore(
    kernels: PortableSparsePanelKernels,
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
            a.rowIdx, a.values, a.colPtr[l], a.colPtr[l + 1], b.data, rows,
        )
    }
}

/**
 * Sparse dtrmv. Direction preserves each source before its destination is written, so no work buffer is
 * needed for the ordinary in-place case.
 *
 * The diagonal is probed as `a[j, j]`, a binary search per column. That is `trmv`'s to pay: `trmm` walks
 * several right-hand sides against one triangle and precomputes the diagonal for itself instead.
 */
internal fun trmvCore(
    kernels: PortableSparsePanelKernels,
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
                kernels.triangularAxpy(j, lower, a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], xj, x)
            }
        }
    } else {
        val order = if (lower) 0 until n else n - 1 downTo 0
        for (j in order) {
            var sum = if (unitDiag) {
                x[j]
            } else {
                a[j, j] * x[j]
            }
            x[j] = kernels.triangularReduce(
                j, lower, a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], x, sum, subtract = false,
            )
        }
    }
}

/** Sparse substitution over RHS panels, so values and indices are read once for several dense columns. */
internal fun trsmLeftCore(
    kernels: PortableSparsePanelKernels,
    a: SparseMatrix,
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean,
    diagonal: DoubleArray?,
    work: DoubleArray,
) {
    val n = a.rows
    val bd = b.data
    val order = if (lower != transpose) 0 until n else n - 1 downTo 0
    forEachRhsPanel(b.cols) { columnStart, width ->
        for (j in order) {
            if (!transpose) {
                val divisor = diagonal?.get(j) ?: 1.0
                kernels.triangularPanelScatter(
                    true, j, lower, diagonal == null, divisor, a.rowIdx, a.values,
                    a.colPtr[j], a.colPtr[j + 1], bd, n, columnStart, width, work,
                )
            } else {
                val divisor = diagonal?.get(j) ?: 1.0
                kernels.triangularPanelGather(
                    true, j, lower, diagonal == null, divisor, a.rowIdx, a.values,
                    a.colPtr[j], a.colPtr[j + 1], bd, n, columnStart, width, work,
                )
            }
        }
    }
}

/** Right solve over contiguous dense columns, which turns every sparse update into a Level 1 operation. */
internal fun trsmRightCore(
    kernels: PortableSparsePanelKernels,
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
            a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], b.data, rows,
        )
    }
}

/** `trsv` over the `n` entries of [x], with the triangle flags resolved once by the caller. */
@Suppress("LongParameterList") // the three BLAS triangle flags
internal fun trsvCore(
    kernels: PortableSparsePanelKernels,
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
            kernels.triangularAxpy(j, lower, a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], -xj, x)
        } else {
            val s = kernels.triangularReduce(
                j, lower, a.rowIdx, a.values, a.colPtr[j], a.colPtr[j + 1], x, x[j], subtract = true,
            )
            x[j] = if (unitDiag) s else s / a[j, j]
        }
    }
}
