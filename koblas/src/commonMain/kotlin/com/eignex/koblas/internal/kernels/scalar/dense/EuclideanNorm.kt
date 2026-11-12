@file:Suppress("NOTHING_TO_INLINE") // Fold primitive arithmetic into the concrete caller before JIT compilation.

package com.eignex.koblas.internal.kernels.scalar.dense

import kotlin.math.abs
import kotlin.math.sqrt

/** Smallest normal double. A squares-sum below this has lost precision to underflow. */
internal const val MIN_NORMAL = 2.2250738585072014e-308

/**
 * Euclidean norm of the run v(off until off + len), using finite-range scaling (BLAS `dnrm2`).
 * When a plain sum of squares overflows or underflows, a second pass factors out the largest magnitude.
 */
internal inline fun euclideanNorm(v: DoubleArray, off: Int, stride: Int, len: Int): Double {
    var s = 0.0
    var iv = off
    for (i in 0 until len) {
        val x = v[iv]
        s += x * x
        iv += stride
    }
    if (s.isFinite() && s >= MIN_NORMAL) return sqrt(s)
    return rescaledEuclideanNorm(v, off, stride, len, s)
}

/** Range protection after a caller has already found an unsafe square sum. */
internal inline fun rescaledEuclideanNorm(v: DoubleArray, off: Int, stride: Int, len: Int, squares: Double): Double =
    rescaledEuclideanNorm(len, squares) { v[off + it * stride] }

/** Inline indexed access keeps range protection shared without gathering or allocating adapters. */
internal inline fun rescaledEuclideanNorm(len: Int, squares: Double, value: (Int) -> Double): Double {
    var maximum = 0.0
    for (i in 0 until len) {
        val magnitude = abs(value(i))
        if (magnitude > maximum) maximum = magnitude
    }
    if (maximum == 0.0 || maximum.isInfinite()) return sqrt(squares)
    var scaledSquares = 0.0
    for (i in 0 until len) {
        val scaled = value(i) / maximum
        scaledSquares += scaled * scaled
    }
    return maximum * sqrt(scaledSquares)
}
