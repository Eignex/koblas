@file:Suppress("LongParameterList") // a diagonal block carries its triangle, its flags and its two strides

package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.hardwareFusedMultiplyAdd
import jdk.incubator.vector.DoubleVector

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
    ) {
        if (size <= 0 || sides <= 0) return
        if (!simdAvailable || rhsStride != 1 || sides < LANE) {
            PortableTriangularKernels.diagonalSolve(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides,
            )
            return
        }
        var lane = 0
        val bound = sides - sides % LANE
        while (lane < bound) {
            solveBlock(t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride)
            lane += LANE
        }
        if (lane < sides) {
            PortableTriangularKernels.diagonalSolve(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride, 1, sides - lane,
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
    ) {
        if (size <= 0 || sides <= 0) return
        if (!simdAvailable || rhsStride != 1 || sides < LANE) {
            PortableTriangularKernels.diagonalMultiply(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides,
            )
            return
        }
        var lane = 0
        val bound = sides - sides % LANE
        while (lane < bound) {
            multiplyBlock(t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride)
            lane += LANE
        }
        if (lane < sides) {
            PortableTriangularKernels.diagonalMultiply(
                t, n, start, size, transposed, lower, unitDiag, x, xOffset + lane, orderStride, 1, sides - lane,
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
    ) {
        for (step in 0 until size) {
            val p = if (lower) step else size - 1 - step
            val at = xOffset + p * orderStride
            var accumulated = DoubleVector.fromArray(SPECIES, x, at)
            val first = if (lower) 0 else p + 1
            val last = if (lower) p else size
            for (q in first until last) {
                val coefficient = triangleEntry(t, n, start, p, q, transposed)
                accumulated = multiplySubtract(
                    DoubleVector.fromArray(SPECIES, x, xOffset + q * orderStride),
                    coefficient,
                    accumulated,
                )
            }
            if (!unitDiag) accumulated = accumulated.div(triangleEntry(t, n, start, p, p, transposed))
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
    ) {
        for (step in 0 until size) {
            val p = if (lower) size - 1 - step else step
            val at = xOffset + p * orderStride
            val own = DoubleVector.fromArray(SPECIES, x, at)
            var accumulated =
                if (unitDiag) own else own.mul(triangleEntry(t, n, start, p, p, transposed))
            val first = if (lower) 0 else p + 1
            val last = if (lower) p else size
            for (q in first until last) {
                val coefficient = triangleEntry(t, n, start, p, q, transposed)
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
