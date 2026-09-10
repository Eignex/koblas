package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.borrowI32

/** Runs [block] against stable sparse values when [destination] aliases the live coefficient buffer. */
internal inline fun <T> withStableSparse(
    a: SparseMatrix,
    destination: DoubleArray,
    workspace: Workspace?,
    block: (SparseMatrix) -> T,
): T {
    if (a.values !== destination) return block(a)
    return workspace.borrow(a.nnz) { copy ->
        a.values.copyInto(copy)
        block(SparseMatrix.wrapTrusted(a.rows, a.cols, a.colPtr, a.rowIdx, copy))
    }
}

/** Runs [block] against stable dense storage when [destination] aliases [b]. */
internal inline fun <T> withStableDense(
    b: DenseMatrix,
    destination: DoubleArray,
    workspace: Workspace?,
    block: (DenseMatrix) -> T,
): T {
    if (b.data !== destination) return block(b)
    return workspace.borrow(b.data.size) { copy ->
        b.data.copyInto(copy)
        block(DenseMatrix.wrap(b.rows, b.cols, copy))
    }
}

/** Immediate alias snapshot for in-place APIs without an explicit workspace. */
@OptIn(UnsafeKoblasApi::class)
internal fun SparseMatrix.stableFor(destination: DoubleArray): SparseMatrix = if (values === destination) {
    SparseMatrix.wrap(rows, cols, colPtr, rowIdx, values.copyOf())
} else {
    this
}

/**
 * Borrows the complete SYRK scratch set as one exception-safe inline scope. No holder object is created;
 * each buffer remains an independent workspace loan and is returned if a later loan or the kernel throws.
 */
@Suppress("LongParameterList")
internal inline fun <T> withSymmetricRankScratch(
    workspace: Workspace?,
    order: Int,
    sourceRows: Int,
    sourceEntries: Int,
    crossinline block: (
        sums: DoubleArray,
        touchedAt: IntArray,
        touched: IntArray,
        rowPointers: IntArray,
        adjacentColumns: IntArray,
        adjacentPositions: IntArray,
        rowCursor: IntArray,
    ) -> T,
): T = workspace.borrow(order) { sums ->
    workspace.borrowI32(order) { touchedAt ->
        workspace.borrowI32(order) { touched ->
            workspace.borrowI32(sourceRows + 1) { rowPointers ->
                workspace.borrowI32(sourceEntries) { adjacentColumns ->
                    workspace.borrowI32(sourceEntries) { adjacentPositions ->
                        workspace.borrowI32(sourceRows) { rowCursor ->
                            block(
                                sums,
                                touchedAt,
                                touched,
                                rowPointers,
                                adjacentColumns,
                                adjacentPositions,
                                rowCursor,
                            )
                        }
                    }
                }
            }
        }
    }
}
