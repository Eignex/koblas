@file:Suppress("LongParameterList") // a diagonal block carries its triangle, its flags and its two strides

package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.dense.DenseTriangularKernels
import com.eignex.koblas.internal.dense.PortableTriangularKernels
import com.eignex.koblas.internal.dense.triangleEntry
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/**
 * JVM Vector API diagonal substitution with portable fallbacks.
 *
 * Lanes cover independent right-hand sides; substitution steps remain a dependency chain.
 * Coefficients and diagonals are broadcast across the sides.
 *
 * Use division directly: a reciprocal of a subnormal diagonal can overflow even when the quotient
 * is finite. Zero and infinite diagonals also retain division semantics.
 *
 * Strided sides and incomplete lane blocks use [PortableTriangularKernels];
 * [implementationsFor] reports those bodies. [gathersRightHandSides] requests a copy when it can
 * amortize vector loads. Species initialization requires the Vector API module, checked by
 * [com.eignex.koblas.BuiltinEngines] before using this backend.
 */
internal object SimdTriangularKernels : DenseTriangularKernels {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = if (simdAvailable) SPECIES.length() else 0

    /** Built once, for the reason [SimdPanelKernels] builds its own once. */
    private val NAME: String = "simd-substitution($LANE lanes)"

    override val name: String get() = NAME

    /**
     * Group right-hand sides so the block repeatedly read during substitution stays resident.
     * This cache grouping is independent of the lane count.
     */
    private const val GROUP_BLOCKS = 4

    override fun rightHandSideGroup(order: Int, sides: Int): Int {
        if (!simdAvailable) return PortableTriangularKernels.rightHandSideGroup(order, sides)
        val recommended = GROUP_BLOCKS * LANE
        if (sides in 1 until recommended) return sides
        return if (sides < 1) 1 else recommended
    }

    /**
     * Gather strided sides when they fill a lane block and the diagonal block has multiple steps.
     * The copy is paid once and reused through the block; fewer sides reach no vector body, and a
     * single step offers no reuse.
     */
    override fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean =
        simdAvailable && rhsStride != 1 && sides >= LANE && order > 1

    override fun implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> = when {
        !simdAvailable -> PortableTriangularKernels.implementationsFor(order, sides, contiguous)
        order <= 0 || sides <= 0 -> emptyList()
        sides == 1 && contiguous -> listOf(NAME)
        !contiguous || sides < LANE -> PortableTriangularKernels.implementationsFor(order, sides, contiguous)
        sides % LANE == 0 -> listOf(NAME)
        else -> listOf(NAME) + PortableTriangularKernels.implementationsFor(order, sides, contiguous)
    }

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
        if (simdAvailable && sides == 1 && orderStride == 1) {
            singleSolve(t, n, start, size, transposed, lower, unitDiag, x, xOffset, tOffset)
            return
        }
        if (!simdAvailable || rhsStride != 1 || sides < LANE) {
            PortableTriangularKernels.diagonalSolve(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
            )
            return
        }
        var lane = 0
        val bound = sides - sides % LANE
        while (lane < bound) {
            solveBlock(t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride, tOffset)
            lane += LANE
        }
        if (lane < sides) {
            PortableTriangularKernels.diagonalSolve(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride,
                1, sides - lane, tOffset,
            )
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
        if (simdAvailable && sides == 1 && orderStride == 1) {
            singleMultiply(t, n, start, size, transposed, lower, unitDiag, x, xOffset, tOffset)
            return
        }
        if (!simdAvailable || rhsStride != 1 || sides < LANE) {
            PortableTriangularKernels.diagonalMultiply(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
            )
            return
        }
        var lane = 0
        val bound = sides - sides % LANE
        while (lane < bound) {
            multiplyBlock(t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride, tOffset)
            lane += LANE
        }
        if (lane < sides) {
            PortableTriangularKernels.diagonalMultiply(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride,
                1, sides - lane, tOffset,
            )
        }
    }

    /**
     * Walk a whole diagonal block per lane block of sides to keep repeatedly read values resident.
     * Interchanging the loops would sweep all sides between successive uses.
     */
    private fun solveBlock(
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
        tOffset: Int,
    ) {
        for (step in 0 until size) {
            val p = if (lower) step else size - 1 - step
            val at = xOffset + p * orderStride
            var accumulated = DoubleVector.fromArray(SPECIES, x, at)
            val first = if (lower) 0 else p + 1
            val last = if (lower) p else size
            for (q in first until last) {
                val coefficient = triangleEntry(t, n, start, p, q, transposed, tOffset)
                accumulated = multiplySubtract(
                    DoubleVector.fromArray(SPECIES, x, xOffset + q * orderStride),
                    coefficient,
                    accumulated,
                )
            }
            if (!unitDiag) accumulated = accumulated.div(triangleEntry(t, n, start, p, p, transposed, tOffset))
            accumulated.intoArray(x, at)
        }
    }

    /** The product counterpart of [solveBlock], over the steps that still hold what they arrived with. */
    private fun multiplyBlock(
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
        tOffset: Int,
    ) {
        for (step in 0 until size) {
            val p = if (lower) size - 1 - step else step
            val at = xOffset + p * orderStride
            val own = DoubleVector.fromArray(SPECIES, x, at)
            var accumulated =
                if (unitDiag) own else own.mul(triangleEntry(t, n, start, p, p, transposed, tOffset))
            val first = if (lower) 0 else p + 1
            val last = if (lower) p else size
            for (q in first until last) {
                val coefficient = triangleEntry(t, n, start, p, q, transposed, tOffset)
                accumulated = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, x, xOffset + q * orderStride),
                    coefficient,
                    accumulated,
                )
            }
            accumulated.intoArray(x, at)
        }
    }

    /**
     * One contiguous right-hand side, vectorised along the block instead of across sides.
     *
     * Untransposed, a column of the block is adjacent in memory, so each finished step is removed from the
     * steps after it in one vector run; transposed, a row is, so each step is one vector dot over the steps
     * before it. As separate one-column panels, every step paid a call and a dispatch for a few lane blocks.
     */
    private fun singleSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        tOffset: Int,
    ) {
        val base = tOffset + start + start * n
        for (step in 0 until size) {
            val p = if (lower) step else size - 1 - step
            val diagonal = base + p + p * n
            if (transposed) {
                // Row p of the block is column start + p of the stored triangle.
                val from = if (lower) 0 else p + 1
                val until = if (lower) p else size
                var value = x[xOffset + p] - runDot(t, base + p * n + from, x, xOffset + from, until - from)
                if (!unitDiag) value /= t[diagonal]
                x[xOffset + p] = value
            } else {
                var value = x[xOffset + p]
                if (!unitDiag) value /= t[diagonal]
                x[xOffset + p] = value
                val from = if (lower) p + 1 else 0
                val until = if (lower) size else p
                runUpdate(t, base + p * n + from, x, xOffset + from, until - from, -value)
            }
        }
    }

    /** The multiply counterpart of [singleSolve], in the order that keeps each input until its last use. */
    private fun singleMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        tOffset: Int,
    ) {
        val base = tOffset + start + start * n
        for (step in 0 until size) {
            val p = if (lower) size - 1 - step else step
            val own = x[xOffset + p]
            val scaled = if (unitDiag) own else t[base + p + p * n] * own
            if (transposed) {
                val from = if (lower) 0 else p + 1
                val until = if (lower) p else size
                x[xOffset + p] = scaled + runDot(t, base + p * n + from, x, xOffset + from, until - from)
            } else {
                x[xOffset + p] = scaled
                val from = if (lower) p + 1 else 0
                val until = if (lower) size else p
                runUpdate(t, base + p * n + from, x, xOffset + from, until - from, own)
            }
        }
    }

    /** `x(i) += scale · t(i)` along one contiguous run, the column step of [singleSolve]. */
    private fun runUpdate(t: DoubleArray, tFrom: Int, x: DoubleArray, xFrom: Int, length: Int, scale: Double) {
        var i = 0
        val bound = SPECIES.loopBound(length)
        if (bound > 0) {
            val coefficient = DoubleVector.broadcast(SPECIES, scale)
            while (i < bound) {
                multiplyAdd(
                    DoubleVector.fromArray(SPECIES, t, tFrom + i),
                    coefficient,
                    DoubleVector.fromArray(SPECIES, x, xFrom + i),
                ).intoArray(x, xFrom + i)
                i += LANE
            }
        }
        while (i < length) {
            x[xFrom + i] += scale * t[tFrom + i]
            i++
        }
    }

    /** The dot of two contiguous runs, the row step of [singleSolve], with two chains on a long run. */
    private fun runDot(t: DoubleArray, tFrom: Int, x: DoubleArray, xFrom: Int, length: Int): Double {
        var i = 0
        var sum = 0.0
        val bound = SPECIES.loopBound(length)
        if (bound > 0) {
            var even = DoubleVector.zero(SPECIES)
            var odd = DoubleVector.zero(SPECIES)
            while (i + 2 * LANE <= bound) {
                even = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, t, tFrom + i),
                    DoubleVector.fromArray(SPECIES, x, xFrom + i),
                    even,
                )
                odd = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, t, tFrom + i + LANE),
                    DoubleVector.fromArray(SPECIES, x, xFrom + i + LANE),
                    odd,
                )
                i += 2 * LANE
            }
            if (i < bound) {
                even = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, t, tFrom + i),
                    DoubleVector.fromArray(SPECIES, x, xFrom + i),
                    even,
                )
                i += LANE
            }
            sum = even.add(odd).reduceLanes(VectorOperators.ADD)
        }
        while (i < length) {
            sum += t[tFrom + i] * x[xFrom + i]
            i++
        }
        return sum
    }

    /**
     * `accumulated - coefficient · values`, fused where the machine has the instruction.
     *
     * Inline because a helper returning a [DoubleVector] that the JIT declines to inline makes the vector a
     * heap object, which would turn the substitution's one live accumulator into an allocation per step of
     * the block. The negated coefficient is what lets the fused form carry the subtraction.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun multiplySubtract(
        values: DoubleVector,
        coefficient: Double,
        accumulated: DoubleVector,
    ): DoubleVector = if (hardwareFusedMultiplyAdd) {
        values.fma(DoubleVector.broadcast(SPECIES, -coefficient), accumulated)
    } else {
        accumulated.sub(values.mul(coefficient))
    }

    /** `accumulated + coefficient · values`, inline for the reason [multiplySubtract] is. */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun multiplyAdd(
        values: DoubleVector,
        coefficient: Double,
        accumulated: DoubleVector,
    ): DoubleVector = if (hardwareFusedMultiplyAdd) {
        values.fma(DoubleVector.broadcast(SPECIES, coefficient), accumulated)
    } else {
        accumulated.add(values.mul(coefficient))
    }
}
