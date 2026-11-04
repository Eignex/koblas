package com.eignex.koblas.internal.sparse

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.sparse.*

/**
 * Borrows the complete rank-update scratch set as one exception-safe inline scope. No holder object is
 * created; each buffer remains an independent arrays loan and is returned if a later loan or the kernel
 * throws.
 */
@Suppress("LongParameterList")
internal inline fun <T> withSymmetricRankScratch(
    arrays: Scratch?,
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
): T = arrays.borrow(order) { sums ->
    arrays.borrowI32(order) { touchedAt ->
        arrays.borrowI32(order) { touched ->
            arrays.borrowI32(scratchLength(sourceRows, "syrk")) { rowPointers ->
                arrays.borrowI32(sourceEntries) { adjacentColumns ->
                    arrays.borrowI32(sourceEntries) { adjacentPositions ->
                        arrays.borrowI32(sourceRows) { rowCursor ->
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
