@file:Suppress("LongParameterList") // a diagonal block carries its triangle, its flags and its two strides

package com.eignex.koblas.internal.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/**
 * The portable diagonal substitution every platform has, and the floor a vector backend falls back to.
 *
 * One right-hand side at a time and one step at a time, which is the definition written out. The step is
 * held in a local while the coefficients before it are removed from it, so a block of order `s` reads each
 * of the other steps once per step rather than writing the destination once per pair.
 *
 * Nothing here is grouped, so [rightHandSideGroup] hands over every right-hand side the caller has, and
 * nothing here is helped by adjacency, so [gathersRightHandSides] declines every copy: the strided loads a
 * right-side call makes are what this backend would make anyway.
 */
internal object PortableTriangularKernels : KoblasEngine() {
    override val implementations = setOf(EngineImplementation.Triangular)

    override val name: String get() = NAME

    override fun rightHandSideGroup(order: Int, sides: Int): Int = if (sides > 0) sides else 1

    override fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean = false

    override fun implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> =
        if (order <= 0 || sides <= 0) emptyList() else listOf(NAME)

    override fun diagonalSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int,
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

    override fun diagonalMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int,
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
    internal fun singleSolve(
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
        tOffset: Int,
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
    internal fun singleMultiply(
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
        tOffset: Int,
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

    private const val NAME: String = "scalar-substitution"
}
