package com.eignex.koblas.sparse.internal

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.borrowI32

/** Immediate alias snapshot for in-place APIs that take no workspace. */
@OptIn(UnsafeKoblasApi::class)
internal fun SparseMatrix.stableFor(destination: DoubleArray): SparseMatrix = if (values === destination) {
    SparseMatrix.wrapTrusted(rows, cols, colPointers, rowIndices, values.copyOf())
} else {
    this
}

/**
 * Borrows the complete rank-update scratch set as one exception-safe inline scope. No holder object is
 * created; each buffer remains an independent workspace loan and is returned if a later loan or the kernel
 * throws.
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
            workspace.borrowI32(scratchLength(sourceRows, "syrk")) { rowPointers ->
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
