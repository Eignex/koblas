package com.eignex.koblas.sparse

import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/** Its own object so the initializer, which touches DoubleVector, runs only once the module is present. */
internal object SparseSimd {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = SPECIES.length()

    val autoScatterEligible: Boolean
        get() = SPECIES.vectorBitSize() == 512 && System.getProperty("os.arch").orEmpty() in X86_ARCHITECTURES

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

    @Suppress("LongParameterList")
    fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        len: Int,
        y: DoubleArray,
    ): Double {
        var k = 0
        val bound = SPECIES.loopBound(len)
        var sum = DoubleVector.zero(SPECIES)
        while (k < bound) {
            val gathered = indexedLoad(y, indices, indexOffset + k)
            sum = DoubleVector.fromArray(SPECIES, values, valueOffset + k).fma(gathered, sum)
            k += LANE
        }
        var s = sum.reduceLanes(VectorOperators.ADD)
        while (k < len) {
            s += values[valueOffset + k] * y[indices[indexOffset + k]]
            k++
        }
        return s
    }

    @Suppress("LongParameterList")
    fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        len: Int,
        y: DoubleArray,
        alpha: Double,
    ) {
        var k = 0
        val bound = SPECIES.loopBound(len)
        val multiplier = DoubleVector.broadcast(SPECIES, alpha)
        while (k < bound) {
            val old = indexedLoad(y, indices, indexOffset + k)
            val increment = DoubleVector.fromArray(SPECIES, values, valueOffset + k)
            // Sparse BLAS updates require the multiplication to round before the addition.
            increment.mul(multiplier).add(old).intoArray(y, 0, indices, indexOffset + k)
            k += LANE
        }
        while (k < len) {
            y[indices[indexOffset + k]] += alpha * values[valueOffset + k]
            k++
        }
    }

    @Suppress("LongParameterList")
    fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        len: Int,
        out: DoubleArray,
    ) {
        var k = 0
        val bound = SPECIES.loopBound(len)
        while (k < bound) {
            DoubleVector.fromArray(SPECIES, values, valueOffset + k).intoArray(out, 0, indices, indexOffset + k)
            k += LANE
        }
        while (k < len) {
            out[indices[indexOffset + k]] = values[valueOffset + k]
            k++
        }
    }

    @Suppress("LongParameterList")
    fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        len: Int,
        from: DoubleArray,
    ) {
        var k = 0
        val bound = SPECIES.loopBound(len)
        while (k < bound) {
            indexedLoad(from, indices, indexOffset + k).intoArray(values, valueOffset + k)
            k += LANE
        }
        while (k < len) {
            values[valueOffset + k] = from[indices[indexOffset + k]]
            k++
        }
    }

    @Suppress("LongParameterList")
    fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        len: Int,
        from: DoubleArray,
    ) {
        var k = 0
        val bound = SPECIES.loopBound(len)
        val zero = DoubleVector.zero(SPECIES)
        while (k < bound) {
            indexedLoad(from, indices, indexOffset + k).intoArray(values, valueOffset + k)
            zero.intoArray(from, 0, indices, indexOffset + k)
            k += LANE
        }
        while (k < len) {
            val index = indices[indexOffset + k]
            values[valueOffset + k] = from[index]
            from[index] = 0.0
            k++
        }
    }

    fun nrm2(indices: IntArray, indexOffset: Int, len: Int, values: DoubleArray): Double {
        var k = 0
        val bound = SPECIES.loopBound(len)
        var sum = DoubleVector.zero(SPECIES)
        while (k < bound) {
            val gathered = indexedLoad(values, indices, indexOffset + k)
            sum = gathered.fma(gathered, sum)
            k += LANE
        }
        var squares = sum.reduceLanes(VectorOperators.ADD)
        while (k < len) {
            val value = values[indices[indexOffset + k]]
            squares += value * value
            k++
        }
        if (squares.isFinite() && squares >= java.lang.Double.MIN_NORMAL) return kotlin.math.sqrt(squares)
        return ScalarIndexedSparseKernels.nrm2(indices, indexOffset, len, values)
    }

    private val X86_ARCHITECTURES = setOf("amd64", "x86_64", "x64")
}
