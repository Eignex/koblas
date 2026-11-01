package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.internal.sparse.IndexedSparseKernels
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/**
 * The indexed dense dot in two lanes. Other operations have no registration here.
 *
 * The dot is the one indexed kernel whose lanes measured faster than HotSpot's scalar loop, and only on two-lane
 * units, where a lane block is two scalar loads put side by side: 0.81 to 0.92 of the scalar time on Apple
 * silicon. An indexed gather or scatter through the Vector API measured slower everywhere else, from about
 * even to 1.4 times on four-lane ARM and from 2 to over 200 times on 512-bit x86, where it fell back to
 * allocating Java code, so gather, scatter, axpy and the indexed norm stay scalar.
 *
 * Its own object so the initializer, which touches DoubleVector, runs only once the module is present.
 */
internal object SparseSimd : IndexedSparseKernels {
    override val name: String = "simd-indexed"
    private val SPECIES = SIMD_SPECIES
    private val LANE = SPECIES.length()

    /** The lane block every kernel here advances by, below which a call is its scalar tail and nothing else. */
    val lanes: Int get() = LANE

    /** Whether this unit takes the vector dot: two lanes, each loaded as a scalar. */
    val vectorDot: Boolean get() = LANE == 2

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
            val gathered = DoubleVector.zero(SPECIES)
                .withLane(0, dense[indices[indexOffset + k]])
                .withLane(1, dense[indices[indexOffset + k + 1]])
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
}
