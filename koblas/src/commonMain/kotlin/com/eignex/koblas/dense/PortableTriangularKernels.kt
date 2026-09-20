@file:Suppress("LongParameterList") // a diagonal block carries its triangle, its flags and its two strides

package com.eignex.koblas.dense

/**
 * The portable diagonal substitution every platform has, and the floor a vector backend falls back to.
 *
 * One right-hand side at a time and one step at a time, which is the definition written out. The step is
 * held in a local while the coefficients before it are removed from it, so a block of order `s` reads each
 * of the other steps once per step rather than writing the destination once per pair.
 *
 * Nothing here is grouped, so [rightHandSideGroup] hands over every right-hand side the caller has: the
 * chunking exists for a backend whose arithmetic wants a fixed number of independent values, and cutting the
 * loop into pieces it has no use for would only cost the calls. Nothing here is helped by adjacency either,
 * so [gathersRightHandSides] declines every copy: the strided loads a right-side call makes are what this
 * backend would make anyway.
 */
internal object PortableTriangularKernels : DenseTriangularKernels {
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
    ) {
        if (size <= 0 || sides <= 0) return
        for (r in 0 until sides) {
            val column = xOffset + r * rhsStride
            for (step in 0 until size) {
                // A lower block is finished from its first step and an upper one from its last, which is the
                // order in which a step's coefficients are already known.
                val p = if (lower) step else size - 1 - step
                var value = x[column + p * orderStride]
                val inner = if (lower) 0 until p else p + 1 until size
                for (q in inner) {
                    value -= triangleEntry(t, n, start, p, q, transposed) * x[column + q * orderStride]
                }
                if (!unitDiag) value /= triangleEntry(t, n, start, p, p, transposed)
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
    ) {
        if (size <= 0 || sides <= 0) return
        for (r in 0 until sides) {
            val column = xOffset + r * rhsStride
            for (step in 0 until size) {
                // Opposite to the solve: a step of a lower block is the sum of the steps at or before it, so
                // the last is formed first and every step it reads still holds what it arrived with.
                val p = if (lower) size - 1 - step else step
                val own = x[column + p * orderStride]
                var value = if (unitDiag) own else triangleEntry(t, n, start, p, p, transposed) * own
                val inner = if (lower) 0 until p else p + 1 until size
                for (q in inner) {
                    value += triangleEntry(t, n, start, p, q, transposed) * x[column + q * orderStride]
                }
                x[column + p * orderStride] = value
            }
        }
    }

    private const val NAME: String = "scalar-substitution"
}
