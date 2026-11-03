@file:Suppress("LongParameterList") // Primitive windows carry arrays, offsets, strides and extents.

package com.eignex.koblas.internal.portable.dense

import com.eignex.koblas.internal.dense.triangleEntry

internal fun scalarDiagonalSolve(
    t: DoubleArray, n: Int, start: Int, size: Int, transposed: Boolean, lower: Boolean, unitDiag: Boolean,
    x: DoubleArray, xOffset: Int, orderStride: Int, rhsStride: Int, sides: Int, tOffset: Int,
) {
    if (size <= 0 || sides <= 0) return
    if (sides == 1) {
        singleSolve(t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, tOffset)
        return
    }
    for (r in 0 until sides) {
        val column = xOffset + r * rhsStride
        for (step in 0 until size) {
            // Substitute in dependency order so each dependent value is already solved.
            val p = if (lower) step else size - 1 - step
            var value = x[column + p * orderStride]
            val inner = if (lower) 0 until p else p + 1 until size
            for (q in inner) {
                value -= triangleEntry(t, n, start, p, q, transposed, tOffset) * x[column + q * orderStride]
            }
            if (!unitDiag) value /= triangleEntry(t, n, start, p, p, transposed, tOffset)
            x[column + p * orderStride] = value
        }
    }
}

internal fun scalarDiagonalMultiply(
    t: DoubleArray, n: Int, start: Int, size: Int, transposed: Boolean, lower: Boolean, unitDiag: Boolean,
    x: DoubleArray, xOffset: Int, orderStride: Int, rhsStride: Int, sides: Int, tOffset: Int,
) {
    if (size <= 0 || sides <= 0) return
    if (sides == 1) {
        singleMultiply(t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, tOffset)
        return
    }
    for (r in 0 until sides) {
        val column = xOffset + r * rhsStride
        for (step in 0 until size) {
            // Multiply in reverse dependency order to preserve input values until their last use.
            val p = if (lower) size - 1 - step else step
            val own = x[column + p * orderStride]
            var value = if (unitDiag) own else triangleEntry(t, n, start, p, p, transposed, tOffset) * own
            val inner = if (lower) 0 until p else p + 1 until size
            for (q in inner) {
                value += triangleEntry(t, n, start, p, q, transposed, tOffset) * x[column + q * orderStride]
            }
            x[column + p * orderStride] = value
        }
    }
}

/**
 * One right-hand side, walked in whichever direction the block is contiguous in. Untransposed, a column
 * of the block is adjacent in memory, so each finished step is removed from the steps after it; transposed,
 * a row is, so each step reduces the steps before it. The general loop reads every coefficient through
 * [triangleEntry] in dot order, which strides across columns on an untransposed block.
 */
private fun singleSolve(
    t: DoubleArray, n: Int, start: Int, size: Int, transposed: Boolean, lower: Boolean, unitDiag: Boolean,
    x: DoubleArray, xOffset: Int, stride: Int, tOffset: Int,
) {
    val base = tOffset + start + start * n
    // M(p, q) is at base + p * rowStep + q * columnStep.
    val rowStep = if (transposed) n else 1
    val columnStep = if (transposed) 1 else n
    for (step in 0 until size) {
        val p = if (lower) step else size - 1 - step
        if (transposed) {
            var value = x[xOffset + p * stride]
            val from = if (lower) 0 else p + 1
            val until = if (lower) p else size
            for (q in from until until) value -= t[base + p * rowStep + q * columnStep] * x[xOffset + q * stride]
            if (!unitDiag) value /= t[base + p * (rowStep + columnStep)]
            x[xOffset + p * stride] = value
        } else {
            var value = x[xOffset + p * stride]
            if (!unitDiag) value /= t[base + p * (rowStep + columnStep)]
            x[xOffset + p * stride] = value
            val from = if (lower) p + 1 else 0
            val until = if (lower) size else p
            val column = base + p * columnStep
            for (q in from until until) x[xOffset + q * stride] -= t[column + q * rowStep] * value
        }
    }
}

/** The multiply counterpart of [singleSolve], in the order that keeps each input until its last use. */
private fun singleMultiply(
    t: DoubleArray, n: Int, start: Int, size: Int, transposed: Boolean, lower: Boolean, unitDiag: Boolean,
    x: DoubleArray, xOffset: Int, stride: Int, tOffset: Int,
) {
    val base = tOffset + start + start * n
    val rowStep = if (transposed) n else 1
    val columnStep = if (transposed) 1 else n
    for (step in 0 until size) {
        val p = if (lower) size - 1 - step else step
        val own = x[xOffset + p * stride]
        if (transposed) {
            var value = if (unitDiag) own else t[base + p * (rowStep + columnStep)] * own
            val from = if (lower) 0 else p + 1
            val until = if (lower) p else size
            for (q in from until until) value += t[base + p * rowStep + q * columnStep] * x[xOffset + q * stride]
            x[xOffset + p * stride] = value
        } else {
            x[xOffset + p * stride] = if (unitDiag) own else t[base + p * (rowStep + columnStep)] * own
            val from = if (lower) p + 1 else 0
            val until = if (lower) size else p
            val column = base + p * columnStep
            for (q in from until until) x[xOffset + q * stride] += t[column + q * rowStep] * own
        }
    }
}
