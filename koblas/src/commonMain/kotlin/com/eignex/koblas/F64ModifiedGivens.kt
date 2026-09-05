package com.eignex.koblas

import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64StridedVectorView
import com.eignex.koblas.core.asView
import com.eignex.koblas.core.overlaps
import kotlin.math.abs

/**
 * A modified Givens transformation produced by [rotmg].
 *
 * [d1], [d2], and [x1] are the state returned by BLAS `drotmg`; [flag] is its `dparam[0]` selector.
 * The four `h` properties always describe the materialized matrix, including entries that BLAS implies
 * from [flag] rather than storing in `dparam`. Thus this type carries the complete transformation without
 * exposing BLAS's unused parameter-array slots.
 *
 * @property d1 updated first scale factor.
 * @property d2 updated second scale factor.
 * @property x1 updated first component.
 * @property flag BLAS modified-rotation selector: `-2.0`, `-1.0`, `0.0`, or `1.0`.
 * @property h11 first-row, first-column entry of the transformation.
 * @property h21 second-row, first-column entry of the transformation.
 * @property h12 first-row, second-column entry of the transformation.
 * @property h22 second-row, second-column entry of the transformation.
 */
public class F64ModifiedGivens internal constructor(
    public val d1: Double,
    public val d2: Double,
    public val x1: Double,
    public val flag: Double,
    public val h11: Double,
    public val h21: Double,
    public val h12: Double,
    public val h22: Double,
)

/**
 * Construct a modified Givens transformation (BLAS `drotmg`) that eliminates the second component of
 * `(sqrt(d1) * x1, sqrt(d2) * y1)`. The returned [F64ModifiedGivens] carries both BLAS's updated
 * `d1`, `d2`, `x1` state and the matrix for [rotm].
 *
 * The scaling constants and branch conditions are the Netlib reference algorithm. In particular, a negative
 * [d1] clears the returned state, and `d2 * y1 == 0.0` returns the identity (`flag == -2.0`) without changing
 * that state.
 */
public fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens =
    koblas.kernels.rotmg(d1, d2, x1, y1)

/** Portable Netlib-reference implementation used by the scalar and fallback kernel backends. */
@Suppress("CyclomaticComplexMethod") // literal translation of the four Netlib DROTMG cases
internal fun portableRotmg(d1: Double, d2: Double, x1: Double, y1: Double): F64ModifiedGivens {
    var dd1 = d1
    var dd2 = d2
    var dx1 = x1
    var flag: Double
    var h11 = 0.0
    var h12 = 0.0
    var h21 = 0.0
    var h22 = 0.0

    if (dd1 < 0.0) {
        flag = -1.0
        dd1 = 0.0
        dd2 = 0.0
        dx1 = 0.0
    } else {
        val p2 = dd2 * y1
        if (p2 == 0.0) return F64ModifiedGivens(dd1, dd2, dx1, -2.0, 1.0, 0.0, 0.0, 1.0)

        val p1 = dd1 * dx1
        val q2 = p2 * y1
        val q1 = p1 * dx1
        if (abs(q1) > abs(q2)) {
            h21 = -y1 / dx1
            h12 = p2 / p1
            val u = 1.0 - h12 * h21
            if (u > 0.0) {
                flag = 0.0
                dd1 /= u
                dd2 /= u
                dx1 *= u
            } else {
                flag = -1.0
                h11 = 0.0
                h12 = 0.0
                h21 = 0.0
                h22 = 0.0
                dd1 = 0.0
                dd2 = 0.0
                dx1 = 0.0
            }
        } else if (q2 < 0.0) {
            flag = -1.0
            h11 = 0.0
            h12 = 0.0
            h21 = 0.0
            h22 = 0.0
            dd1 = 0.0
            dd2 = 0.0
            dx1 = 0.0
        } else {
            flag = 1.0
            h11 = p1 / p2
            h22 = dx1 / y1
            val u = 1.0 + h11 * h22
            val temp = dd2 / u
            dd2 = dd1 / u
            dd1 = temp
            dx1 = y1 * u
        }

        if (dd1 != 0.0) {
            // Scaling by GAM/GAMSQ never brings an infinite dd1 back into range, so require finiteness too;
            // otherwise a non-finite d1 or d2 input spins this loop forever instead of returning.
            while (dd1.isFinite() && (dd1 <= R_GAMSQ || dd1 >= GAMSQ)) {
                // Netlib guards FIX-H on a non-negative flag. Once the flag is -1 the implied entries have
                // already been materialized and carry scaled values, so writing the literals again would
                // discard every rescaling step but the last.
                if (flag >= 0.0) {
                    if (flag == 0.0) {
                        h11 = 1.0
                        h22 = 1.0
                    } else {
                        h21 = -1.0
                        h12 = 1.0
                    }
                    flag = -1.0
                }
                if (dd1 <= R_GAMSQ) {
                    dd1 *= GAMSQ
                    dx1 /= GAM
                    h11 /= GAM
                    h12 /= GAM
                } else {
                    dd1 /= GAMSQ
                    dx1 *= GAM
                    h11 *= GAM
                    h12 *= GAM
                }
            }
        }
        if (dd2 != 0.0) {
            while (dd2.isFinite() && (abs(dd2) <= R_GAMSQ || abs(dd2) >= GAMSQ)) {
                if (flag >= 0.0) {
                    if (flag == 0.0) {
                        h11 = 1.0
                        h22 = 1.0
                    } else {
                        h21 = -1.0
                        h12 = 1.0
                    }
                    flag = -1.0
                }
                if (abs(dd2) <= R_GAMSQ) {
                    dd2 *= GAMSQ
                    h21 /= GAM
                    h22 /= GAM
                } else {
                    dd2 /= GAMSQ
                    h21 *= GAM
                    h22 *= GAM
                }
            }
        }
    }

    return modifiedGivens(dd1, dd2, dx1, flag, h11, h21, h12, h22)
}

/**
 * Apply a modified Givens [transformation] to each pair in [x] and [y] (BLAS `drotm`). Both vectors are
 * overwritten in place; when they share a backing array, the [F64StridedVectorView] overload's overlap
 * handling applies, so at a shared physical entry the final write is from [y].
 */
public fun rotm(x: F64DenseVector, y: F64DenseVector, transformation: F64ModifiedGivens) {
    rotm(x.asView(), y.asView(), transformation)
}

/**
 * [rotm] over borrowed strided storage. Negative strides are supported. If the views overlap, both logical
 * input sequences are snapshotted before writing; at a shared physical entry the final write is from [y].
 */
public fun rotm(x: F64StridedVectorView, y: F64StridedVectorView, transformation: F64ModifiedGivens) {
    requireSameSize(x.size, y.size)
    if (transformation.flag == -2.0) return
    if (x.overlaps(y)) {
        val snapshotX = x.toDoubleArray()
        val snapshotY = y.toDoubleArray()
        applyModifiedGivens(snapshotX, 0, 1, snapshotY, 0, 1, x.size, transformation)
        for (i in 0 until x.size) x[i] = snapshotX[i]
        for (i in 0 until y.size) y[i] = snapshotY[i]
    } else {
        koblas.kernels.rotm(
            x.data,
            x.offset,
            x.stride,
            y.data,
            y.offset,
            y.stride,
            x.size,
            transformation,
        )
    }
}

/** Convert the materialized matrix back to BLAS's flag-dependent `dparam` layout. */
internal fun F64ModifiedGivens.toBlasParameters(): DoubleArray = DoubleArray(5).also { parameters ->
    parameters[0] = flag
    when (flag) {
        -2.0 -> {
            // BLAS never reads dparam[1..4] for the identity flag, so nothing to populate.
        }

        -1.0 -> {
            parameters[1] = h11
            parameters[2] = h21
            parameters[3] = h12
            parameters[4] = h22
        }

        0.0 -> {
            parameters[2] = h21
            parameters[3] = h12
        }

        1.0 -> {
            parameters[1] = h11
            parameters[4] = h22
        }

        else -> error("unexpected modified Givens flag $flag")
    }
}

private fun modifiedGivens(
    d1: Double,
    d2: Double,
    x1: Double,
    flag: Double,
    h11: Double,
    h21: Double,
    h12: Double,
    h22: Double,
): F64ModifiedGivens = when (flag) {
    // portableRotmg returns the -2.0 identity directly, before reaching this factory.
    -1.0 -> F64ModifiedGivens(d1, d2, x1, flag, h11, h21, h12, h22)

    // BLAS fixes these off-diagonal entries at 1.0/-1.0 for flag 0/1; h21/h12 (flag 0) and h11/h22
    // (flag 1) are the only entries flag leaves for the caller to have computed.
    0.0 -> F64ModifiedGivens(d1, d2, x1, flag, 1.0, h21, h12, 1.0)

    1.0 -> F64ModifiedGivens(d1, d2, x1, flag, h11, -1.0, 1.0, h22)

    else -> error("unexpected modified Givens flag $flag")
}

/** Portable backend implementation of modified Givens application. */
@Suppress("LongParameterList")
internal fun portableRotm(
    x: DoubleArray,
    xOff: Int,
    xStride: Int,
    y: DoubleArray,
    yOff: Int,
    yStride: Int,
    len: Int,
    transformation: F64ModifiedGivens,
) {
    if (transformation.flag == -2.0) return
    applyModifiedGivens(x, xOff, xStride, y, yOff, yStride, len, transformation)
}

/**
 * Apply [transformation] in place to the strided [x]/[y] runs, loading each pair before writing either
 * result, so this is safe even when [x] and [y] are the same array at the same offset and stride.
 */
@Suppress("LongParameterList")
internal fun applyModifiedGivens(
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
    len: Int,
    transformation: F64ModifiedGivens,
) {
    val h11 = transformation.h11
    val h12 = transformation.h12
    val h21 = transformation.h21
    val h22 = transformation.h22
    for (i in 0 until len) {
        val xi = x[xOffset + i * xStride]
        val yi = y[yOffset + i * yStride]
        x[xOffset + i * xStride] = h11 * xi + h12 * yi
        y[yOffset + i * yStride] = h21 * xi + h22 * yi
    }
}

private const val GAM: Double = 4096.0
private const val GAMSQ: Double = 16777216.0
private const val R_GAMSQ: Double = 5.9604645e-8
