package com.eignex.koblas

import kotlin.math.abs
import kotlin.math.sqrt

/** Smallest normal double; a squares-sum below this has lost precision to underflow. */
private const val MIN_NORMAL = 2.2250738585072014e-308

/**
 * `sqrt(Sum v[off..off+len-1]²)`, accurate over the whole double range (BLAS `dnrm2`).
 *
 * The fast path is a plain sum of squares; when that overflows or drowns in underflow — components beyond
 * roughly `1e±150` — a second pass factors out the largest magnitude so the squares stay in range. Any
 * finite input therefore yields the correct norm, which netlib `dnrm2` also guarantees and a bare
 * `sqrt(sum)` does not.
 *
 * Shared by the dense and sparse level-1 kernels, which need exactly the same three passes over a
 * contiguous run: for a `SparseVector` that run is its value array, since the unstored entries contribute
 * nothing to a sum of squares. Having it once means the two cannot drift apart on the rescaling threshold
 * or on which of `0.0`, `NaN` and `Inf` fall through the fast path.
 */
internal fun euclideanNorm(v: DoubleArray, off: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) {
        val x = v[off + i]
        s += x * x
    }
    if (s.isFinite() && s >= MIN_NORMAL) return sqrt(s)
    var amax = 0.0
    for (i in 0 until len) {
        val a = abs(v[off + i])
        if (a > amax) amax = a
    }
    // All-zero (0.0), NaN anywhere (NaN), or an infinite component (Inf) resolve through the raw sum.
    if (amax == 0.0 || amax.isInfinite()) return sqrt(s)
    var t = 0.0
    for (i in 0 until len) {
        val r = v[off + i] / amax
        t += r * r
    }
    return amax * sqrt(t)
}
