package com.eignex.koblas

import kotlin.math.abs
import kotlin.math.sqrt

/** Smallest normal double. A squares-sum below this has lost precision to underflow. */
internal const val F64_MIN_NORMAL = 2.2250738585072014e-308

/**
 * Euclidean norm of the run v(off until off + len), accurate over the whole double range (BLAS `dnrm2`).
 * When a plain sum of squares overflows or underflows, a second pass factors out the largest magnitude.
 */
internal fun euclideanNorm(v: DoubleArray, off: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) {
        val x = v[off + i]
        s += x * x
    }
    if (s.isFinite() && s >= F64_MIN_NORMAL) return sqrt(s)
    var amax = 0.0
    for (i in 0 until len) {
        val a = abs(v[off + i])
        if (a > amax) amax = a
    }
    if (amax == 0.0 || amax.isInfinite()) return sqrt(s)
    var t = 0.0
    for (i in 0 until len) {
        val r = v[off + i] / amax
        t += r * r
    }
    return amax * sqrt(t)
}

/** Sum of absolute values over the run v(off until off + len) (BLAS `dasum`). */
internal fun absoluteSum(v: DoubleArray, off: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) s += abs(v[off + i])
    return s
}
