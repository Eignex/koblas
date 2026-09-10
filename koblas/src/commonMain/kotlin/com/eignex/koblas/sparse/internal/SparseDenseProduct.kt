@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C

package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.dense.borrowTransposed
import com.eignex.koblas.sparse.PortableSparsePanelKernels
import com.eignex.koblas.sparse.REFERENCE_SPARSE_RHS_WIDTH
import kotlin.math.min

/** Visits dense right-hand sides in cache-sized panels. */
internal inline fun forEachRhsPanel(columns: Int, action: (start: Int, width: Int) -> Unit) {
    var start = 0
    while (start < columns) {
        val width = min(REFERENCE_SPARSE_RHS_WIDTH, columns - start)
        action(start, width)
        start += width
    }
}

@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheLeft(
    kernels: PortableSparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    m: Int,
    n: Int,
    k: Int,
    workspace: Workspace?,
) {
    val leadingDimension = b.rows
    workspace.borrow(REFERENCE_SPARSE_RHS_WIDTH) { work ->
        forEachRhsPanel(n) { columnStart, width ->
            if (transposeA) {
                for (outputRow in 0 until m) {
                    kernels.gatherProductPanel(
                        alpha, outputRow, a.rowIdx, a.values, a.colPtr[outputRow], a.colPtr[outputRow + 1],
                        b.data, leadingDimension, transposeB, columnStart, width, c.data, m, work,
                    )
                }
            } else {
                for (inner in 0 until k) {
                    kernels.scatterProductPanel(
                        alpha, a.rowIdx, a.values, a.colPtr[inner], a.colPtr[inner + 1], b.data,
                        leadingDimension, transposeB, inner, columnStart, width, c.data, m, work,
                    )
                }
            }
        }
    }
}

/** `C += alpha · op(B) · op(A)` over the sparse operand's CSC columns. */
@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheRight(
    kernels: PortableSparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    m: Int,
    workspace: Workspace?,
) {
    if (transposeB) {
        workspace.borrowTransposed(b.data, b.rows, b.cols) { packed ->
            multiplyFromTheRightColumns(kernels, alpha, a, transposeA, c.data, packed, m, m)
        }
    } else {
        multiplyFromTheRightColumns(kernels, alpha, a, transposeA, c.data, b.data, m, b.rows)
    }
}

internal fun multiplyFromTheRightColumns(
    kernels: PortableSparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    c: DoubleArray,
    b: DoubleArray,
    rows: Int,
    leadingDimension: Int,
) {
    for (column in 0 until a.cols) {
        kernels.rightProductColumn(
            alpha, column, a.rowIdx, a.values, a.colPtr[column], a.colPtr[column + 1], transposeA,
            b, leadingDimension, c, rows,
        )
    }
}
