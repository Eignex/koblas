@file:Suppress("LongParameterList") // a diagonal block carries its triangle, its flags and its two strides

package com.eignex.koblas.dense

// A triangular route is checked against the diagonal blocks a call really substituted; the recorder below
// delegates the arithmetic, so a call under it computes the same answer while saying what it was handed.

/** One diagonal block a triangular routine substituted, as the window and strides it was handed. */
internal class DiagonalCall(
    val start: Int,
    val size: Int,
    val transposed: Boolean,
    val lower: Boolean,
    val unitDiag: Boolean,
    val orderStride: Int,
    val rhsStride: Int,
    val sides: Int,
    val solve: Boolean,
)

/** A substitution backend that records every diagonal block it is handed and delegates the arithmetic. */
internal class RecordingTriangles(private val delegate: DenseTriangularKernels) : DenseTriangularKernels {
    /** Every diagonal block, in call order. */
    val blocks: MutableList<DiagonalCall> = ArrayList()

    override val name: String get() = "recording(${delegate.name})"

    override fun rightHandSideGroup(order: Int, sides: Int): Int = delegate.rightHandSideGroup(order, sides)

    override fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean =
        delegate.gathersRightHandSides(rhsStride, order, sides)

    override fun implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> =
        delegate.implementationsFor(order, sides, contiguous)

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
        blocks.add(
            DiagonalCall(start, size, transposed, lower, unitDiag, orderStride, rhsStride, sides, solve = true),
        )
        delegate.diagonalSolve(
            t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides,
        )
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
        blocks.add(
            DiagonalCall(start, size, transposed, lower, unitDiag, orderStride, rhsStride, sides, solve = false),
        )
        delegate.diagonalMultiply(
            t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides,
        )
    }
}
