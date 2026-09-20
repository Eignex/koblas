@file:Suppress("LongParameterList") // a diagonal block carries its triangle, its flags and its two strides

package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.hardwareFusedMultiplyAdd
import jdk.incubator.vector.DoubleVector

/**
 * The JVM Vector API diagonal substitution, with the portable one underneath it.
 *
 * What is vectorised here is the one thing about a triangular routine that is parallel: the right-hand
 * sides. The substitution down a block's order is a dependency chain and stays one, step after step; across
 * the sides nothing depends on anything, so a lane block of them is one step of that chain done at once.
 * Every side in a lane block divides by the same diagonal and subtracts the same coefficient, so the
 * coefficient is broadcast and the sides are loaded, which is the opposite of what a vectorised
 * matrix-vector product does with the same triangle.
 *
 * The division is a division. Substituting a reciprocal multiply would be the usual way to get a vector
 * divide out of a loop and it is not done: a solve is asked about singular and near-singular triangles more
 * than about any other kind, and a reciprocal changes the answer exactly there. A subnormal diagonal has no
 * finite reciprocal, so the multiply gives an infinity where the division gives a finite number, and the
 * result of dividing by a zero or an infinity is what the caller asked to see.
 *
 * Two things fall through to [PortableTriangularKernels], and [implementationsFor] names them rather than
 * letting this object's name stand for them. Right-hand sides the caller left strided cannot be loaded as a
 * lane block at all, which is why [gathersRightHandSides] asks for a copy where one is worth making and why
 * a call that declines it runs the portable body. And the sides a last lane block does not fill are portable
 * as well, at most one lane block short of the whole.
 *
 * Resolving the species is what initializing this costs, so a runtime without the module must not reach it:
 * [com.eignex.koblas.BuiltinEngines] offers no engine holding this backend there.
 */
internal object SimdTriangularKernels : DenseTriangularKernels {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = if (simdAvailable) SPECIES.length() else 0

    /** Built once, for the reason [SimdPanelKernels] builds its own once. */
    private val NAME: String = "simd-substitution($LANE lanes)"

    override val name: String get() = NAME

    /**
     * Lane blocks of right-hand sides one substitution walks before moving to the next.
     *
     * The block of sides a substitution is working on is read once per step for every step before it, so
     * the whole of it wants to stay resident: a diagonal block of sixty-four steps by this many sides is a
     * few kilobytes, which is the size at which that means something. It is a grouping and not a lane
     * count, and the two are different numbers that agree only by coincidence.
     */
    private const val GROUP_BLOCKS = 4

    override fun rightHandSideGroup(order: Int, sides: Int): Int {
        if (!simdAvailable) return PortableTriangularKernels.rightHandSideGroup(order, sides)
        val recommended = GROUP_BLOCKS * LANE
        // Never more than the call has, which is this contract's own bound rather than something a caller
        // is left to impose: a recommendation wider than the work is not a recommendation about the work.
        if (sides in 1 until recommended) return sides
        return if (sides < 1) 1 else recommended
    }

    /**
     * Strided right-hand sides are worth gathering once there are enough of them to fill a lane block.
     *
     * The copy is two passes over one diagonal block's worth of the caller's data and it buys every step of
     * that block a vector load instead of a strided one, so it is paid once and read back `order` times.
     * Below one lane block there is no vector body to reach and the copy would buy nothing; at one step
     * there is nothing to read back and the substitution is a division the caller could have done in place.
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
     * One lane block of right-hand sides substituted through the whole diagonal block.
     *
     * The lane block is the outer loop on purpose. Every step reads each step before it, so the piece of the
     * block being worked on is read `size` times over; walking the whole order for one lane block at a time
     * is what keeps that piece where it was last touched, and walking every lane block per step would sweep
     * the whole width between one use and the next.
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
