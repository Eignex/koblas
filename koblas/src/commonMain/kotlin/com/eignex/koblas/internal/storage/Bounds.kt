package com.eignex.koblas.internal.storage

import com.eignex.koblas.*

/**
 * A vector window of [count] entries from [offset], [stride] apart, which a routine reads or writes as
 * `values[offset + i * stride]`. Both ends are checked, so a negative stride is accepted when it stays inside.
 */
internal fun requireVectorWindow(bufferSize: Int, offset: Int, count: Int, stride: Int, what: String) {
    requireShape(count >= 0) { "$what: negative length $count" }
    require(stride != 0) { "$what: stride must not be zero" }
    if (count == 0) return
    val last = offset.toLong() + (count - 1).toLong() * stride
    val first = minOf(offset.toLong(), last)
    val end = maxOf(offset.toLong(), last)
    requireShape(first >= 0 && end < bufferSize) {
        "$what: $count entries from $offset with stride $stride address [$first, $end] outside length $bufferSize"
    }
}

/** `require` for a shape check, throwing [DimensionMismatch] instead of a bare `IllegalArgumentException`. */
internal inline fun requireShape(condition: Boolean, message: () -> String) {
    if (!condition) throw DimensionMismatch(message())
}

/** `require` for an index check, so every storage reports the same standard type. */
internal inline fun requireIndex(condition: Boolean, message: () -> String) {
    if (!condition) throw IndexOutOfBoundsException(message())
}

/** Rejects a shape no storage can hold, before an allocation turns it into an arithmetic error. */
internal fun requireNonNegativeShape(rows: Int, cols: Int) {
    requireShape(rows >= 0 && cols >= 0) { "negative shape: ${rows}x$cols" }
}

/** Bounds for a vector position. The message builds only on the failing path, since [requireIndex] inlines. */
internal fun requireInBounds(i: Int, size: Int) {
    requireIndex(i in 0 until size) { "index $i outside [0,$size)" }
}

/** Bounds for a matrix position, both indices at once so one failure names the pair. */
internal fun requireInBounds(i: Int, j: Int, rows: Int, cols: Int) {
    requireIndex(i in 0 until rows && j in 0 until cols) { "index ($i;$j) outside ${rows}x$cols" }
}
