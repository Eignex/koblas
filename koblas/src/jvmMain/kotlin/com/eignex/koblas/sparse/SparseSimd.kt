package com.eignex.koblas.sparse

import com.eignex.koblas.internal.numeric.hardwareFusedMultiplyAdd
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/** Its own object so the initializer, which touches DoubleVector, runs only once the module is present. */
internal object SparseSimd : IndexedSparseKernels {
    override val name: String = "simd"

    // Rescaling depends on values; sparse intersections always use the scalar merge.
    override fun implementationFor(operation: SparseOperation, count: Int): String? = when (operation) {
        SparseOperation.IndexedNrm2 -> null
        SparseOperation.DotSparse -> ScalarIndexedSparseKernels.name
        else -> name
    }

    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = SPECIES.length()

    /** The lane block every kernel here advances by, below which a call is its scalar tail and nothing else. */
    val lanes: Int get() = LANE

    val autoScatterEligible: Boolean
        get() = SPECIES.vectorBitSize() == 512 && System.getProperty("os.arch").orEmpty() in X86_ARCHITECTURES

    // AVX2 gather loses to HotSpot's scalar indexed loop in the arithmetic gather cases.
    val autoGatherEligible: Boolean
        get() = autoIndexedLoadEligible &&
            (SPECIES.vectorBitSize() != 256 || System.getProperty("os.arch").orEmpty() !in X86_ARCHITECTURES)

    val autoIndexedLoadEligible: Boolean
        get() = LANE > 1

    // NEON has no gather instruction; the Vector API indexed load allocates fallback carriers on JDK 25.
    // Inline the two scalar loads so the vector stays inside the caller's C2 compilation.
    @Suppress("NOTHING_TO_INLINE")
    private inline fun indexedLoad(values: DoubleArray, indices: IntArray, offset: Int): DoubleVector = if (LANE == 2) {
        DoubleVector.zero(SPECIES)
            .withLane(0, values[indices[offset]])
            .withLane(1, values[indices[offset + 1]])
    } else {
        DoubleVector.fromArray(SPECIES, values, 0, indices, offset)
    }

    /**
     * One multiply-add, fused where [hardwareFusedMultiplyAdd] found the instruction and two operations
     * where it did not.
     *
     * Inline for the reason [indexedLoad] gives, and because the test is then a constant the JIT folds, so
     * one of the two forms reaches the loop and neither a branch nor the other form survives in it.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun multiplyAdd(x: DoubleVector, y: DoubleVector, accumulator: DoubleVector): DoubleVector =
        if (hardwareFusedMultiplyAdd) x.fma(y, accumulator) else x.mul(y).add(accumulator)

    @Suppress("LongParameterList")
    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double {
        var k = 0
        val bound = SPECIES.loopBound(count)
        var sum = DoubleVector.zero(SPECIES)
        while (k < bound) {
            val gathered = indexedLoad(dense, indices, indexOffset + k)
            sum = multiplyAdd(DoubleVector.fromArray(SPECIES, values, valueOffset + k), gathered, sum)
            k += LANE
        }
        var s = sum.reduceLanes(VectorOperators.ADD)
        while (k < count) {
            s += values[valueOffset + k] * dense[indices[indexOffset + k]]
            k++
        }
        return s
    }

    @Suppress("LongParameterList")
    override fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ) {
        var k = 0
        val bound = SPECIES.loopBound(count)
        val multiplier = DoubleVector.broadcast(SPECIES, alpha)
        while (k < bound) {
            val old = indexedLoad(destination, indices, indexOffset + k)
            val increment = DoubleVector.fromArray(SPECIES, values, valueOffset + k)
            // Not [multiplyAdd], which the reductions use. A product that overflows to infinity stays
            // infinite once it has rounded, where a fused one carries it and can land back in range, so the
            // two disagree about whether a result exists at all rather than about its last bit. This loop
            // also has an indexed load and an indexed store around every operation, so fusing the arithmetic
            // between them buys nothing worth that.
            increment.mul(multiplier).add(old).intoArray(destination, 0, indices, indexOffset + k)
            k += LANE
        }
        while (k < count) {
            destination[indices[indexOffset + k]] += alpha * values[valueOffset + k]
            k++
        }
    }

    @Suppress("LongParameterList")
    override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        var k = 0
        val bound = SPECIES.loopBound(count)
        while (k < bound) {
            DoubleVector.fromArray(SPECIES, values, valueOffset + k).intoArray(destination, 0, indices, indexOffset + k)
            k += LANE
        }
        while (k < count) {
            destination[indices[indexOffset + k]] = values[valueOffset + k]
            k++
        }
    }

    @Suppress("LongParameterList")
    override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        var k = 0
        val bound = SPECIES.loopBound(count)
        while (k < bound) {
            indexedLoad(source, indices, indexOffset + k).intoArray(values, valueOffset + k)
            k += LANE
        }
        while (k < count) {
            values[valueOffset + k] = source[indices[indexOffset + k]]
            k++
        }
    }

    @Suppress("LongParameterList")
    override fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        var k = 0
        val bound = SPECIES.loopBound(count)
        val zero = DoubleVector.zero(SPECIES)
        while (k < bound) {
            indexedLoad(source, indices, indexOffset + k).intoArray(values, valueOffset + k)
            zero.intoArray(source, 0, indices, indexOffset + k)
            k += LANE
        }
        while (k < count) {
            val index = indices[indexOffset + k]
            values[valueOffset + k] = source[index]
            source[index] = 0.0
            k++
        }
    }

    override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
        var k = 0
        val bound = SPECIES.loopBound(count)
        var sum = DoubleVector.zero(SPECIES)
        while (k < bound) {
            val gathered = indexedLoad(values, indices, indexOffset + k)
            // Squaring is a multiply-add like any other, so it takes the same route.
            sum = multiplyAdd(gathered, gathered, sum)
            k += LANE
        }
        var squares = sum.reduceLanes(VectorOperators.ADD)
        while (k < count) {
            val value = values[indices[indexOffset + k]]
            squares += value * value
            k++
        }
        if (squares.isFinite() && squares >= java.lang.Double.MIN_NORMAL) return kotlin.math.sqrt(squares)
        return ScalarIndexedSparseKernels.nrm2(indices, indexOffset, count, values)
    }

    @Suppress("LongParameterList")
    override fun dotSparse(
        xIndices: IntArray,
        xIndexOffset: Int,
        xValues: DoubleArray,
        xValueOffset: Int,
        xCount: Int,
        yIndices: IntArray,
        yIndexOffset: Int,
        yValues: DoubleArray,
        yValueOffset: Int,
        yCount: Int,
    ): Double = ScalarIndexedSparseKernels.dotSparse(
        xIndices, xIndexOffset, xValues, xValueOffset, xCount,
        yIndices, yIndexOffset, yValues, yValueOffset, yCount,
    )

    private val X86_ARCHITECTURES = setOf("amd64", "x86_64", "x64")
}
