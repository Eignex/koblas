package com.eignex.koblas.dense

import com.eignex.koblas.StridedVectorView
import com.eignex.koblas.VectorLike
import com.eignex.koblas.forEachStored
import kotlin.math.abs

/** Ordered dot product over two strided vector views. */
internal fun stridedDot(left: StridedVectorView, right: StridedVectorView): Double {
    var sum = 0.0
    for (index in 0 until left.size) {
        sum += left.data[left.offset + index * left.stride] * right.data[right.offset + index * right.stride]
    }
    return sum
}

/** Ordered dot product through the general vector access contract. */
internal fun genericDot(left: VectorLike, right: VectorLike): Double {
    var sum = 0.0
    for (index in 0 until left.size) sum += left[index] * right[index]
    return sum
}

/** Ordered sum over represented entries. */
internal fun storedSum(vector: VectorLike): Double {
    var sum = 0.0
    vector.forEachStored { _, value -> sum += value }
    return sum
}

/** Ordered Neumaier sum over represented entries. */
internal fun storedCompensatedSum(vector: VectorLike): Double {
    var sum = 0.0
    var compensation = 0.0
    vector.forEachStored { _, value ->
        val total = sum + value
        compensation += if (abs(sum) >= abs(value)) (sum - total) + value else (value - total) + sum
        sum = total
    }
    return sum + compensation
}

/** Ordered sum of absolute represented entries. */
internal fun storedAbsoluteSum(vector: VectorLike): Double {
    var sum = 0.0
    vector.forEachStored { _, value -> sum += abs(value) }
    return sum
}

/** First represented index with maximum absolute value, or zero when no strict maximum was seen. */
internal fun storedIndexOfMaximumAbsoluteValue(vector: VectorLike): Int {
    var best = -1
    var bestAbsolute = 0.0
    vector.forEachStored { index, value ->
        val absolute = abs(value)
        if (absolute > bestAbsolute) {
            bestAbsolute = absolute
            best = index
        }
    }
    return if (best == -1) 0 else best
}

/** Copies a general vector to a strided destination. */
internal fun copyToStrided(source: VectorLike, destination: StridedVectorView) {
    for (index in 0 until source.size) destination[index] = source[index]
}

/** Copies represented entries into an already zeroed dense destination. */
internal fun copyStoredToDense(source: VectorLike, destination: DoubleArray) {
    source.forEachStored { index, value -> destination[index] = value }
}

/** Exchanges two strided vectors in logical order. */
internal fun swapStrided(left: StridedVectorView, right: StridedVectorView) {
    for (index in 0 until left.size) {
        val value = left[index]
        left[index] = right[index]
        right[index] = value
    }
}

/** Adds represented entries from a general vector to a dense destination. */
internal fun addStoredToDense(destination: DoubleArray, alpha: Double, source: VectorLike) {
    source.forEachStored { index, value -> destination[index] += alpha * value }
}

/** Adds a scaled general vector to a strided destination. */
internal fun addToStrided(destination: StridedVectorView, alpha: Double, source: VectorLike) {
    for (index in 0 until destination.size) destination[index] += alpha * source[index]
}

/** Scales a strided vector in logical order. */
internal fun scaleStridedVector(vector: StridedVectorView, alpha: Double) {
    for (index in 0 until vector.size) vector[index] *= alpha
}

/** Stable Euclidean norm over strided storage, including NaN propagation. */
internal fun stridedEuclideanNorm(vector: StridedVectorView): Double {
    var scale = 0.0
    var sumSquares = 1.0
    for (index in 0 until vector.size) {
        val value = abs(vector[index])
        if (value != 0.0) {
            if (scale < value) {
                val ratio = scale / value
                sumSquares = 1.0 + sumSquares * ratio * ratio
                scale = value
            } else {
                val ratio = value / scale
                sumSquares += ratio * ratio
            }
        }
    }
    if (scale == 0.0) return if (sumSquares.isNaN()) Double.NaN else 0.0
    return scale * kotlin.math.sqrt(sumSquares)
}
