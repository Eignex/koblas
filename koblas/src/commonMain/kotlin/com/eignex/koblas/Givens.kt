@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * @property c the cosine.
 * @property s the sine.
 * @property r the rotated length `±hypot(a, b)`, what `a` becomes as `b` goes to zero.
 */
public class Givens internal constructor(public val c: Double, public val s: Double, public val r: Double)

/**
 * Generate the plane rotation that zeroes [b] against [a] (BLAS `drotg`), rescaling so squares that would
 * overflow or vanish still rotate correctly. Netlib sign convention; the all-zero pair gives the identity.
 */
public fun rotg(a: Double, b: Double): Givens {
    if (b == 0.0 && a == 0.0) return Givens(c = 1.0, s = 0.0, r = 0.0)
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
    return Givens(c = a / r, s = b / r, r = r)
}

/**
 * Apply a plane rotation (BLAS `drot`). Each pair `(x_i, y_i)` becomes `(c*x_i + s*y_i, c*y_i - s*x_i)`,
 * so both [x] and [y] are overwritten in place. Dense vectors use the same overlap semantics as the
 * borrowed overload.
 */
public fun rot(x: DenseVector, y: DenseVector, rotation: Givens) {
    rot(x.asView(), y.asView(), rotation)
}

/**
 * [rot] over borrowed strided storage. Negative strides are supported. If the views overlap, both logical
 * input sequences are snapshotted before writing; at a shared physical entry the final write is from [y].
 */
public fun rot(x: StridedVectorView, y: StridedVectorView, rotation: Givens) {
    requireSameSize(x.size, y.size)
    if (rotation.c == 1.0 && rotation.s == 0.0) return
    if (x.overlaps(y)) {
        val snapshotX = x.toDoubleArray()
        val snapshotY = y.toDoubleArray()
        portableRot(snapshotX, 0, snapshotY, 0, x.size, rotation.c, rotation.s)
        for (i in 0 until x.size) x[i] = snapshotX[i]
        for (i in 0 until y.size) y[i] = snapshotY[i]
    } else if (x.stride == 1 && y.stride == 1) {
        koblas.vectorKernels.rot(x.data, x.offset, y.data, y.offset, x.size, rotation.c, rotation.s)
    } else {
        for (i in 0 until x.size) {
            val xi = x[i]
            val yi = y[i]
            x[i] = rotation.c * xi + rotation.s * yi
            y[i] = rotation.c * yi - rotation.s * xi
        }
    }
}

/** Portable backend implementation of plane rotation application. */
@Suppress("LongParameterList")
@kotlin.jvm.JvmSynthetic
internal fun portableRot(x: DoubleArray, xOff: Int, y: DoubleArray, yOff: Int, len: Int, c: Double, s: Double) {
    for (i in 0 until len) {
        val xi = x[xOff + i]
        val yi = y[yOff + i]
        x[xOff + i] = c * xi + s * yi
        y[yOff + i] = c * yi - s * xi
    }
}
