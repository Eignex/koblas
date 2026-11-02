package com.eignex.koblas.internal.sparse

internal fun indexedSquareSum(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
    // Independent sums avoid serialising every indexed load behind one addition chain.
    var squares = 0.0
    var squares1 = 0.0
    var squares2 = 0.0
    var squares3 = 0.0
    var k = 0
    val bound = count - 3
    while (k < bound) {
        val value = values[indices[indexOffset + k]]
        val value1 = values[indices[indexOffset + k + 1]]
        val value2 = values[indices[indexOffset + k + 2]]
        val value3 = values[indices[indexOffset + k + 3]]
        squares += value * value
        squares1 += value1 * value1
        squares2 += value2 * value2
        squares3 += value3 * value3
        k += 4
    }
    squares += squares1 + squares2 + squares3
    while (k < count) {
        val value = values[indices[indexOffset + k]]
        squares += value * value
        k++
    }
    return squares
}
