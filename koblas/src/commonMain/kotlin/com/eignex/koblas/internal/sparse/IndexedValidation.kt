package com.eignex.koblas.internal.sparse

import com.eignex.koblas.*

internal fun validateColumnDot(matrix: SparseMatrix, column: Int, dense: DoubleArray) {
    requireIndex(column in 0 until matrix.cols) { "column $column is outside [0, ${matrix.cols})" }
    requireShape(matrix.rows == dense.size) { "dotColumn: sizes differ, ${matrix.rows} vs ${dense.size}" }
}

internal fun requireSparseWindow(length: Int, offset: Int, count: Int, name: String) {
    require(offset >= 0 && count >= 0 && offset.toLong() + count <= length) {
        "$name window [$offset, ${offset.toLong() + count}) exceeds length $length"
    }
}

internal fun validateSparseIndices(
    indices: IntArray,
    offset: Int,
    count: Int,
    dimension: Int,
    strictlyIncreasing: Boolean,
) {
    var previous = -1
    for (k in 0 until count) {
        val index = indices[offset + k]
        requireIndex(index in 0 until dimension) { "index $index is outside [0, $dimension)" }
        if (strictlyIncreasing) {
            require(index > previous) { "indices must be strictly increasing" }
            previous = index
        }
    }
}
