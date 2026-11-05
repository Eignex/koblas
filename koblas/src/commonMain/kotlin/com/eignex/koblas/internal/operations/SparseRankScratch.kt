package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.borrowI32
import com.eignex.koblas.internal.storage.sparse.scratchLength

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
