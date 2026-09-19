package com.eignex.koblas.sparse.internal

import com.eignex.koblas.requireShape

/**
 * Checked arithmetic for the array lengths a CSC result needs.
 *
 * A validated [com.eignex.koblas.SparseMatrix] already bounds its own column count, because holding
 * `cols + 1` pointers is what makes it a matrix. Its row count is bounded by nothing, so an operation whose
 * output has one column per input row is where an extent that cannot be represented first becomes an array
 * length. Reaching that as a negative length is an arithmetic accident; reaching it as a shape error is the
 * answer the operation owes its caller.
 */
internal fun pointerLength(columns: Int, what: String): Int {
    requireShape(columns < Int.MAX_VALUE) {
        "$what: a CSC result with $columns columns needs ${columns.toLong() + 1} pointers, more than one array can hold"
    }
    return columns + 1
}

/**
 * The same check for scratch indexed by an extent rather than by a result's columns.
 *
 * An operation that discovers nothing still has to say whether it could have: a rank update over an empty
 * source needs no row scratch at all, and the ones that do need it are rejected here rather than by the
 * allocator.
 */
internal fun scratchLength(extent: Int, what: String): Int {
    requireShape(extent < Int.MAX_VALUE) {
        "$what: scratch for $extent rows needs ${extent.toLong() + 1} entries, more than one array can hold"
    }
    return extent + 1
}
