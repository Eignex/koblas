package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.internal.kernels.KernelProvider
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/**
 * Only two-lane indexed dots measured faster than scalar loops. Indexed gather, scatter and axpy stay scalar:
 * Vector API indexed operations measured slower and can fall back to allocating Java code.
 * Keep a separate object so DoubleVector initialization occurs only when the module is present.
 */
internal object SimdIndexedKernels : KernelProvider {
    override val id: String = "jvm-simd"
    override fun minimumLength(implementation: EngineImplementation): Int = 2

    override val name: String = "simd-indexed"
    private val SPECIES = SIMD_SPECIES
    private val LANE = SPECIES.length()

    /** The lane block every kernel here advances by, below which a call is its scalar tail and nothing else. */
    val lanes: Int get() = LANE

    /** Whether this unit takes the vector dot: two lanes, each loaded as a scalar. */
    val vectorDot: Boolean get() = LANE == 2

    override val implementations = if (vectorDot) setOf(EngineImplementation.IndexedDot) else emptySet()

    @Suppress("LongParameterList")
    override fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double {
        var k = 0
        // The body fills exactly two lanes, so a wider unit runs the scalar tail alone rather than drop entries.
        val bound = if (vectorDot) SPECIES.loopBound(count) else 0
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
