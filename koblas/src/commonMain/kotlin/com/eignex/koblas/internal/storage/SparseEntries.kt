package com.eignex.koblas.internal.storage

/** Copies the entries outside [start] until [end], leaving [count] replacement slots for [write]. */
internal inline fun spliceSparseEntries(
    indices: IntArray,
    values: DoubleArray,
    start: Int,
    end: Int,
    count: Int,
    write: (IntArray, DoubleArray) -> Unit,
) {
    val size = values.size.toLong() - (end - start) + count
    requireShape(size <= Int.MAX_VALUE) { "stored entry count exceeds Int capacity" }
    val newIndices = IntArray(size.toInt())
    val newValues = DoubleArray(size.toInt())
    indices.copyInto(newIndices, 0, 0, start)
    values.copyInto(newValues, 0, 0, start)
    indices.copyInto(newIndices, start + count, end)
    values.copyInto(newValues, start + count, end)
    write(newIndices, newValues)
}
