@file:Suppress("LongParameterList") // Primitive windows carry arrays, offsets, strides and extents.

package com.eignex.koblas.internal.portable.dense

/**
 * One diagonal block substituted for each right-hand side in turn, walked in whichever direction the block is
 * contiguous in. Untransposed, a column of the block is adjacent in memory, so each finished step is removed
 * from the steps after it; transposed, a row is, so each step reduces the steps before it.
 */
internal fun scalarDiagonalSolve(
    t: DoubleArray,
    n: Int,
    start: Int,
    size: Int,
    transposed: Boolean,
    lower: Boolean,
    unitDiag: Boolean,
    x: DoubleArray,
    xOffset: Int,
    stride: Int,
    rhsStride: Int,
    sides: Int,
    tOffset: Int,
) {
    val base = tOffset + start + start * n
    // M(p, q) is at base + p * rowStep + q * columnStep.
    val rowStep = if (transposed) n else 1
    val columnStep = if (transposed) 1 else n
    for (r in 0 until sides) {
        val origin = xOffset + r * rhsStride
        for (step in 0 until size) {
            val p = if (lower) step else size - 1 - step
            if (transposed) {
                var value = x[origin + p * stride]
                val from = if (lower) 0 else p + 1
                val until = if (lower) p else size
                for (q in from until until) value -= t[base + p * rowStep + q * columnStep] * x[origin + q * stride]
                if (!unitDiag) value /= t[base + p * (rowStep + columnStep)]
                x[origin + p * stride] = value
            } else {
                var value = x[origin + p * stride]
                if (!unitDiag) value /= t[base + p * (rowStep + columnStep)]
                x[origin + p * stride] = value
                val from = if (lower) p + 1 else 0
                val until = if (lower) size else p
                val column = base + p * columnStep
                for (q in from until until) x[origin + q * stride] -= t[column + q * rowStep] * value
            }
        }
    }
}

/** The multiply counterpart of [scalarDiagonalSolve], in the order that keeps each input until its last use. */
internal fun scalarDiagonalMultiply(
    t: DoubleArray,
    n: Int,
    start: Int,
    size: Int,
    transposed: Boolean,
    lower: Boolean,
    unitDiag: Boolean,
    x: DoubleArray,
    xOffset: Int,
    stride: Int,
    rhsStride: Int,
    sides: Int,
    tOffset: Int,
) {
    val base = tOffset + start + start * n
    val rowStep = if (transposed) n else 1
    val columnStep = if (transposed) 1 else n
    for (r in 0 until sides) {
        val origin = xOffset + r * rhsStride
        for (step in 0 until size) {
            val p = if (lower) size - 1 - step else step
            val own = x[origin + p * stride]
            if (transposed) {
                var value = if (unitDiag) own else t[base + p * (rowStep + columnStep)] * own
                val from = if (lower) 0 else p + 1
                val until = if (lower) p else size
                for (q in from until until) value += t[base + p * rowStep + q * columnStep] * x[origin + q * stride]
                x[origin + p * stride] = value
            } else {
                x[origin + p * stride] = if (unitDiag) own else t[base + p * (rowStep + columnStep)] * own
                val from = if (lower) p + 1 else 0
                val until = if (lower) size else p
                val column = base + p * columnStep
                for (q in from until until) x[origin + q * stride] += t[column + q * rowStep] * own
            }
        }
    }
}

/** `M(p, q)` for a diagonal block starting at [start], which is the entry a substitution reads. */
internal fun triangleEntry(
    t: DoubleArray,
    n: Int,
    start: Int,
    p: Int,
    q: Int,
    transposed: Boolean,
    offset: Int = 0,
): Double = if (transposed) {
    t[offset + start + q + (start + p) * n]
} else {
    t[offset + start + p + (start + q) * n]
}
