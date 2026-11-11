package com.eignex.koblas.internal.storage.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.internal.storage.requireShape

/** Grows with stored entries rather than the logical domain, without boxing indices or coefficients. */
internal class SparseEntries {
    private var indices = IntArray(0)
    private var values = DoubleArray(0)
    var size: Int = 0
        private set

    fun add(index: Int, value: Double) {
        if (size == indices.size) {
            requireShape(size < Int.MAX_VALUE) { "stored entry count exceeds Int capacity" }
            val capacity = maxOf(16L, size.toLong() * 2).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
            indices = indices.copyOf(capacity)
            values = values.copyOf(capacity)
        }
        indices[size] = index
        values[size] = value
        size++
    }

    /** Hands off unique ascending entries; the builder must not be used after handing off storage. */
    fun toVector(length: Int): SparseVector = SparseVector.wrap(length, finishedIndices(), finishedValues())

    /** Hands off CSC entries already ordered within each column, with no sorting or merging. */
    fun toMatrix(rows: Int, columns: Int, pointers: IntArray): SparseMatrix =
        SparseMatrix.wrapTrusted(rows, columns, pointers, finishedIndices(), finishedValues())

    private fun finishedIndices(): IntArray = if (size == indices.size) indices else indices.copyOf(size)

    private fun finishedValues(): DoubleArray = if (size == values.size) values else values.copyOf(size)
}
