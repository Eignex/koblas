@file:Suppress("LongParameterList") // a diagonal block carries its triangle, its flags and its two strides

package com.eignex.koblas.dense

import com.eignex.koblas.internal.dense.DenseTriangularKernels

// Recording diagonal substitutions checks routes against executed blocks while retaining their arithmetic.

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
internal class RecordingTriangles(private val delegate: DenseTriangularKernels) : DenseTriangularKernels by delegate {
    /** Every diagonal block, in call order. */
    val blocks: MutableList<DiagonalCall> = ArrayList()

    override val name: String get() = "recording(${delegate.name})"

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
        blocks.add(
            DiagonalCall(start, size, transposed, lower, unitDiag, orderStride, rhsStride, sides, solve = true),
        )
        delegate.diagonalSolve(
            t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
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
        tOffset: Int,
    ) {
        blocks.add(
            DiagonalCall(start, size, transposed, lower, unitDiag, orderStride, rhsStride, sides, solve = false),
        )
        delegate.diagonalMultiply(
            t, n, start, size, transposed, lower, unitDiag, x, xOffset, orderStride, rhsStride, sides, tOffset,
        )
    }
}
