@file:Suppress("LongParameterList") // Primitive arithmetic windows avoid per-call descriptors.

package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*

/** Diagonal substitution arithmetic and right-hand-side layout recommendations. */
internal interface TriangularKernels : KernelProvider {
    /** The bodies a diagonal block of [order] steps over [sides] right-hand sides reaches. */
    fun implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> =
        if (order <= 0 || sides <= 0) emptyList() else listOf(name)

    /**
     * How many right-hand sides this engine recommends substituting at once, for a block of [order] steps
     * over [sides] of them.
     *
     * Always at least one and never more than [sides] when that is positive. It is a grouping and not a lane
     * count, and it is independent of the register tile, the panel grouping and the diagonal block the
     * scheduling chose: a diagonal block is as wide as the dependency chain is long, and this is how much of
     * the work beside it is done together.
     */
    fun rightHandSideGroup(order: Int, sides: Int): Int

    /**
     * Whether a group of right-hand sides this far apart is worth copying into adjacent storage first.
     *
     * The copy is two passes over a block of the caller's data, out and back, and it buys a substitution
     * whose independent values lie next to each other. An engine whose arithmetic is a value at a time gains
     * nothing from that and answers false, and so does one whose right-hand sides are already adjacent.
     * Where this answers true the caller gathers.
     */
    fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean

    /**
     * Solves `M · X = X` in place over one diagonal block, for every right-hand side at once.
     *
     * Substitutes forward when [lower] and backward when it is not, dividing each finished step by its
     * diagonal coefficient unless [unitDiag]. No singularity check or particular result for singular or
     * non-finite inputs is promised.
     */
    fun diagonalSolve(
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
        tOffset: Int = 0,
    ): Unit

    /**
     * Multiplies `X = M · X` in place over one diagonal block, for every right-hand side at once.
     *
     * The order is the one that leaves no step needed after it has been overwritten, which runs opposite to
     * the solve's: a lower block is substituted from its last step upward, because a step is the sum of the
     * steps at or before it and those must still hold what they arrived with.
     */
    fun diagonalMultiply(
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
        tOffset: Int = 0,
    ): Unit
}
