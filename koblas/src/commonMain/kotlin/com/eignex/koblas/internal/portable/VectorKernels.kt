package com.eignex.koblas.internal.portable

import com.eignex.koblas.*
import kotlin.math.abs

/** Sparse support bounds both the reads and the work, even when the other vector has opaque storage. */
internal fun portableDot(x: Vector, y: Vector): Double {
    var sum = 0.0
    if (x is SparseVector) {
        for (k in x.indices.indices) sum += x.values[k] * y[x.indices[k]]
    } else {
        for (i in 0 until x.size) sum += x[i] * y[i]
    }
    return sum
}

internal fun portableSum(x: Vector): Double {
    var sum = 0.0
    for (i in 0 until x.size) sum += x[i]
    return sum
}

internal fun portableAbsoluteSum(x: Vector): Double {
    var sum = 0.0
    for (i in 0 until x.size) sum += abs(x[i])
    return sum
}

internal fun portableMaxAbs(x: Vector): Int {
    if (x.size == 0) return -1
    var best = 0
    var bestAbs = 0.0
    x.forEachStored { i, value ->
        val magnitude = abs(value)
        if (magnitude > bestAbs) {
            bestAbs = magnitude
            best = i
        }
    }
    return best
}

/** Assignment is required here: scaling by zero need not erase nonfinite entries or negative zeros. */
internal fun zeroVector(values: DoubleArray, offset: Int, size: Int, stride: Int) {
    if (stride == 1) {
        values.fill(0.0, offset, offset + size)
    } else {
        for (i in 0 until size) values[offset + i * stride] = 0.0
    }
}
