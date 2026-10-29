package com.eignex.koblas.internal.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.borrowI32
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

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

/** An independent copy of a validated CSC matrix, retaining its stored zeros and coefficient values. */
@Suppress("NOTHING_TO_INLINE") // Keep snapshot construction in the caller without an extra preparation call.
internal inline fun SparseMatrix.sparseCopy(): SparseMatrix = SparseMatrix.wrapTrusted(
    rows,
    cols,
    copyColumnPointers(),
    copyRowIndices(),
    values.copyOf(),
)
