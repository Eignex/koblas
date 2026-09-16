package com.eignex.koblas.internal.numeric

import kotlin.math.abs

/*
 * Each run is walked by an induction variable rather than by multiplying the index, which is the shape the
 * reference BLAS uses and keeps a stride from costing a multiply per element. A stride may be negative; the
 * caller decides which end of its storage the logical first entry sits at.
 */

/** First index of the largest magnitude in the run, ignoring NaNs and retaining the first tie. */
internal fun scalarIamax(v: DoubleArray, vOff: Int, vStride: Int, len: Int): Int {
    if (len == 0) return -1
    var best = 0
    var bestAbs = 0.0
    var iv = vOff
    for (i in 0 until len) {
        val magnitude = abs(v[iv])
        if (magnitude > bestAbs) {
            bestAbs = magnitude
            best = i
        }
        iv += vStride
    }
    return best
}

/** `a · b` over [len] elements of each run. */
@Suppress("LongParameterList") // two strided runs
internal fun scalarDot(
    a: DoubleArray,
    aOff: Int,
    aStride: Int,
    b: DoubleArray,
    bOff: Int,
    bStride: Int,
    len: Int,
): Double {
    var s = 0.0
    var ia = aOff
    var ib = bOff
    for (i in 0 until len) {
        s += a[ia] * b[ib]
        ia += aStride
        ib += bStride
    }
    return s
}

/** Exchange the two runs over [len] elements. */
@Suppress("LongParameterList") // two strided runs
internal fun scalarSwap(a: DoubleArray, aOff: Int, aStride: Int, b: DoubleArray, bOff: Int, bStride: Int, len: Int) {
    var ia = aOff
    var ib = bOff
    for (i in 0 until len) {
        val t = a[ia]
        a[ia] = b[ib]
        b[ib] = t
        ia += aStride
        ib += bStride
    }
}

/** `y += alpha * x` over [len] elements. A zero [alpha] leaves y alone rather than adding zeroes. */
@Suppress("LongParameterList") // two strided runs plus the multiplier
internal fun scalarAxpy(
    y: DoubleArray,
    yOff: Int,
    yStride: Int,
    alpha: Double,
    x: DoubleArray,
    xOff: Int,
    xStride: Int,
    len: Int,
) {
    if (alpha == 0.0) return
    var iy = yOff
    var ix = xOff
    for (i in 0 until len) {
        y[iy] += alpha * x[ix]
        iy += yStride
        ix += xStride
    }
}

/** `v *= alpha` over [len] elements. A unit [alpha] leaves v alone. */
internal fun scalarScale(v: DoubleArray, vOff: Int, vStride: Int, alpha: Double, len: Int) {
    if (alpha == 1.0) return
    var iv = vOff
    for (i in 0 until len) {
        v[iv] *= alpha
        iv += vStride
    }
}
