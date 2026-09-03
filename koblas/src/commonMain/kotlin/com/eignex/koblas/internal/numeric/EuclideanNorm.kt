package com.eignex.koblas.internal.numeric

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

/** Plain sum over the run v(off until off + len). Not a BLAS routine: `dasum` sums absolute values. */
internal fun scalarSum(v: DoubleArray, off: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) s += v[off + i]
    return s
}

/**
 * Sum over the run v(off until off + len) with Neumaier's compensation, which tracks the low-order bits each
 * addition drops and adds them back at the end. A naive sum of `n` terms carries an error growing with `n`,
 * so a long run's total says less than its digits suggest; this holds to within one rounding of the exact
 * sum whatever the length or the ordering.
 *
 * Neumaier's variant rather than plain Kahan because it compensates when the running total is smaller in
 * magnitude than the term being added, which is the case a naive sum handles worst.
 */
internal fun neumaierSum(v: DoubleArray, off: Int, len: Int): Double {
    var s = 0.0
    var compensation = 0.0
    for (i in 0 until len) {
        val x = v[off + i]
        val t = s + x
        compensation += if (abs(s) >= abs(x)) (s - t) + x else (x - t) + s
        s = t
    }
    return s + compensation
}

/** Sum of absolute values over the run v(off until off + len) (BLAS `dasum`). */
internal fun absoluteSum(v: DoubleArray, off: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) s += abs(v[off + i])
    return s
}
