@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C

package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.MatrixWorkspace
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.borrow
import com.eignex.koblas.sparse.SPARSE_RHS_WIDTH
import com.eignex.koblas.sparse.SparsePanelKernels
import kotlin.math.min

/** Visits dense right-hand sides in panels, so one walk of a sparse column serves several of them. */
internal inline fun forEachRhsPanel(columns: Int, action: (start: Int, width: Int) -> Unit) {
    var start = 0
    while (start < columns) {
        val width = min(SPARSE_RHS_WIDTH, columns - start)
        action(start, width)
        start += width
    }
}

/** `C += alpha · op(A) · op(B)` for a sparse `A` on the left, reusing each walk over a small RHS panel. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheLeft(
    kernels: SparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    m: Int,
    n: Int,
    k: Int,
    workspace: MatrixWorkspace?,
) {
    val leadingDimension = b.rows
    workspace.borrow(SPARSE_RHS_WIDTH) { work ->
        forEachRhsPanel(n) { columnStart, width ->
            if (transposeA) {
                for (outputRow in 0 until m) {
                    kernels.gatherProductPanel(
                        alpha, outputRow, a.rowIndices, a.values,
                        a.colPointers[outputRow], a.colPointers[outputRow + 1],
                        b.values, leadingDimension, transposeB, columnStart, width, c.values, m, work,
                    )
                }
            } else {
                for (inner in 0 until k) {
                    kernels.scatterProductPanel(
                        alpha, a.rowIndices, a.values, a.colPointers[inner], a.colPointers[inner + 1],
                        b.values, leadingDimension, transposeB, inner, columnStart, width, c.values, m, work,
                    )
                }
            }
        }
    }
}

/**
 * `C += alpha · op(B) · op(A)` over the sparse operand's CSC columns.
 *
 * With the sparse operand on the right every update is a whole column of the dense block, so a transposed
 * dense operand is staged once into column-major order rather than read across its rows once per entry.
 */
@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheRight(
    kernels: SparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    m: Int,
    workspace: MatrixWorkspace?,
) {
    if (transposeB) {
        workspace.borrow(b.values.size) { packed ->
            for (j in 0 until b.cols) {
                for (i in 0 until b.rows) packed[j + i * b.cols] = b.values[i + j * b.rows]
            }
            multiplyFromTheRightColumns(kernels, alpha, a, transposeA, c.values, packed, m, m)
        }
    } else {
        multiplyFromTheRightColumns(kernels, alpha, a, transposeA, c.values, b.values, m, b.rows)
    }
}

@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, their flags, and both dense windows
internal fun multiplyFromTheRightColumns(
    kernels: SparsePanelKernels,
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
            alpha, column, a.rowIndices, a.values, a.colPointers[column], a.colPointers[column + 1], transposeA,
            b, leadingDimension, c, rows,
        )
    }
}
