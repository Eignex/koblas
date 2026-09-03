package com.eignex.koblas

import com.eignex.koblas.core.F64DenseVector
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * @property c the cosine.
 * @property s the sine.
 * @property r the rotated length `±hypot(a, b)`, what `a` becomes as `b` goes to zero.
 */
public class F64Givens internal constructor(public val c: Double, public val s: Double, public val r: Double)

/**
 * Generate the plane rotation that zeroes [b] against [a] (BLAS `drotg`), rescaling so squares that would
 * overflow or vanish still rotate correctly. Netlib sign convention; the all-zero pair gives the identity.
 */
public fun rotg(a: Double, b: Double): F64Givens {
    if (b == 0.0 && a == 0.0) return F64Givens(c = 1.0, s = 0.0, r = 0.0)
    val absA = abs(a)
    val absB = abs(b)
    val scale = maxOf(absA, absB)
    val ra = a / scale
    val rb = b / scale
    val magnitude = scale * sqrt(ra * ra + rb * rb)
    val r = if (absA > absB) {
        if (a >= 0.0) magnitude else -magnitude
    } else {
        if (b >= 0.0) magnitude else -magnitude
    }
    return F64Givens(c = a / r, s = b / r, r = r)
}

/**
 * Apply a plane rotation (BLAS `drot`). Each pair `(x_i, y_i)` becomes `(c*x_i + s*y_i, c*y_i - s*x_i)`,
 * so both [x] and [y] are overwritten in place.
 */
public fun rot(x: F64DenseVector, y: F64DenseVector, rotation: F64Givens) {
    requireSameSize(x.size, y.size)
    if (rotation.c == 1.0 && rotation.s == 0.0) return
    koblas.kernels.rot(x.data, 0, y.data, 0, x.size, rotation.c, rotation.s)
}

/** Portable backend implementation of plane rotation application. */
@Suppress("LongParameterList")
internal fun portableRot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
    for (i in 0 until len) {
        val xi = x[xOff + i]
        val yi = y[yOff + i]
        x[xOff + i] = c * xi + s * yi
        y[yOff + i] = c * yi - s * xi
    }
}
