package com.eignex.koblas.internal.kernels.scalar

import com.eignex.koblas.*
import kotlin.math.abs

/** Sparse support bounds both the reads and the work, even when the other vector has opaque storage. */
internal fun scalarVectorDot(x: Vector, y: Vector): Double {
    var sum = 0.0
    x.forEachStored { i, value -> sum += value * y[i] }
    return sum
}

internal fun scalarVectorSum(x: Vector): Double {
    var sum = 0.0
    x.forEachStored { _, value -> sum += value }
    return sum
}

internal fun scalarVectorAbsoluteSum(x: Vector): Double {
    var sum = 0.0
    x.forEachStored { _, value -> sum += abs(value) }
    return sum
}

internal fun scalarVectorMaximumMagnitude(x: Vector): Double {
    var maximum = 0.0
    x.forEachStored { _, value ->
        val magnitude = abs(value)
        if (magnitude > maximum) maximum = magnitude
    }
    return maximum
}

internal fun scalarVectorScale(alpha: Double, x: Vector) {
    x.forEachStored { i, value -> x[i] = value * alpha }
}

/** The caller stages [x] before entering this loop when its storage may alias [y]. */
internal fun scalarVectorAxpy(alpha: Double, x: Vector, y: Vector) {
    x.forEachStored { i, value -> y[i] += alpha * value }
}
