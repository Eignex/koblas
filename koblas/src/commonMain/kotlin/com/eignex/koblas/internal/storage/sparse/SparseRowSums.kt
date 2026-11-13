package com.eignex.koblas.internal.storage.sparse

/**
 * Primitive sums over touched nonnegative rows, with storage proportional to their distinct support.
 *
 * CSC row indices leave `-1` available as an empty slot, so neither keys nor sums need boxed entries.
 * A half-full table always leaves room for linear probing, independently of the logical row domain.
 */
internal class SparseRowSums {
    private var rows = IntArray(16) { -1 }
    private var sums = DoubleArray(16)
    private var size = 0

    fun add(row: Int, value: Double) {
        require(row >= 0) { "row must be nonnegative" }
        var slot = slotOf(row)
        if (rows[slot] < 0) {
            if (size == rows.size / 2) {
                grow()
                slot = slotOf(row)
            }
            rows[slot] = row
            size++
        }
        sums[slot] += value
    }

    /** Largest accumulated sum, including the zero of an absent row. */
    fun maximumOrZero(): Double {
        var result = 0.0
        for (sum in sums) result = maxOf(result, sum)
        return result
    }

    private fun slotOf(row: Int): Int {
        val hash = row * -1640531527
        val mask = rows.size - 1
        var slot = (hash xor (hash ushr 16)) and mask
        while (rows[slot] >= 0 && rows[slot] != row) slot = (slot + 1) and mask
        return slot
    }

    private fun grow() {
        check(rows.size < (1 shl 30)) { "sparse row sums exceed array capacity" }
        val oldRows = rows
        val oldSums = sums
        rows = IntArray(oldRows.size * 2) { -1 }
        sums = DoubleArray(rows.size)
        for (slot in oldRows.indices) {
            val row = oldRows[slot]
            if (row < 0) continue
            val destination = slotOf(row)
            rows[destination] = row
            sums[destination] = oldSums[slot]
        }
    }
}
