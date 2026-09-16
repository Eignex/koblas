@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

/**
 * Apply a plane rotation (BLAS `drot`). Each pair `(x_i, y_i)` becomes `(c*x_i + s*y_i, c*y_i - s*x_i)`,
 * so both [x] and [y] are overwritten in place. Dense vectors use the same overlap semantics as the
 * borrowed overload.
 *
 * The rotation arrives as its cosine and sine rather than through a generator. BLAS `drotg` produced those
 * two from a pair of scalars, along with the rotated length only it consumed, and is not part of Koblas:
 * `c = a / hypot(a, b)` and `s = b / hypot(a, b)` is the whole of it, and a caller who wants a different sign
 * convention than Netlib's, which is what LAPACK's own `dlartg` exists to provide, would have to write it out
 * anyway.
 */
public fun rot(x: DenseVector, y: DenseVector, c: Double, s: Double) {
    rot(x.asView(), y.asView(), c, s)
}

/**
 * [rot] over borrowed strided storage. Negative strides are supported. If the views overlap, both logical
 * input sequences are snapshotted before writing; at a shared physical entry the final write is from [y].
 */
public fun rot(x: StridedVector, y: StridedVector, c: Double, s: Double) {
    requireSameSize(x.size, y.size)
    if (c == 1.0 && s == 0.0) return
    if (x.overlaps(y)) {
        val snapshotX = x.toDoubleArray()
        val snapshotY = y.toDoubleArray()
        portableRot(snapshotX, 0, snapshotY, 0, x.size, c, s)
        for (i in 0 until x.size) x[i] = snapshotX[i]
        for (i in 0 until y.size) y[i] = snapshotY[i]
    } else if (x.stride == 1 && y.stride == 1) {
        koblas.vectorKernels.rot(x.data, x.offset, y.data, y.offset, x.size, c, s)
    } else {
        for (i in 0 until x.size) {
            val xi = x[i]
            val yi = y[i]
            x[i] = c * xi + s * yi
            y[i] = c * yi - s * xi
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
