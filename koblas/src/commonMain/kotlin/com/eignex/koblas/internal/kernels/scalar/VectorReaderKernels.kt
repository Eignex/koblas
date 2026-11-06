package com.eignex.koblas.internal.kernels.scalar

import com.eignex.koblas.*
import kotlin.math.abs

/** Sparse support bounds both the reads and the work, even when the other vector has opaque storage. */
internal fun scalarVectorDot(x: Vector, y: Vector): Double {
    var sum = 0.0
    if (x is SparseVector) {
        for (k in x.offset until x.offset + x.nnz) sum += x.values[k] * y[x.indices[k]]
    } else {
        for (i in 0 until x.size) sum += x[i] * y[i]
    }
    return sum
}

internal fun scalarVectorSum(x: Vector): Double {
    var sum = 0.0
    for (i in 0 until x.size) sum += x[i]
    return sum
}

internal fun scalarVectorAbsoluteSum(x: Vector): Double {
    var sum = 0.0
    for (i in 0 until x.size) sum += abs(x[i])
    return sum
}

internal fun scalarVectorMaximumMagnitude(x: Vector): Double {
    var maximum = 0.0
    for (i in 0 until x.size) {
        val magnitude = abs(x[i])
        if (magnitude > maximum) maximum = magnitude
    }
    return maximum
}

internal fun scalarVectorScale(alpha: Double, x: Vector) {
    for (i in 0 until x.size) x[i] *= alpha
}

/** The caller stages [x] before entering this loop when its storage may alias [y]. */
internal fun scalarVectorAxpy(alpha: Double, x: Vector, y: Vector) {
    if (x is SparseVector) {
        for (k in x.offset until x.offset + x.nnz) y[x.indices[k]] += alpha * x.values[k]
    } else {
        for (i in 0 until x.size) y[i] += alpha * x[i]
    }
}

/** Assignment is required here: scaling by zero need not erase nonfinite entries or negative zeros. */
internal fun zeroVector(values: DoubleArray, offset: Int, size: Int, stride: Int) {
    if (stride == 1) {
        values.fill(0.0, offset, offset + size)
    } else {
        for (i in 0 until size) values[offset + i * stride] = 0.0
    }
}
