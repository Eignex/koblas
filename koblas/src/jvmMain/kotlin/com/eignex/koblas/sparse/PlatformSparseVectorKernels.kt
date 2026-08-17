package com.eignex.koblas.sparse

import com.eignex.koblas.BackendNames
import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.requireShape
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/** The nonzero count from which the gathered usdot beats the portable loop. */
private const val SPARSE_DOT_SIMD_MIN = 128

/** A gathered usdot above [SPARSE_DOT_SIMD_MIN], the portable loops below. */
internal actual object PlatformSparseVectorKernels : SparseVectorKernels {
    actual override val name: String get() = if (simdAvailable) BackendNames.SIMD_SPARSE else BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** Computes usdot with the dense operand gathered a register at a time. */
    actual override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        if (!simdAvailable || x.indices.size < SPARSE_DOT_SIMD_MIN) return super.dot(x, y)
        return SparseSimd.dot(x.indices, x.values, y)
    }
}

/** Its own object so the initializer, which touches DoubleVector, runs only once the module is present. */
private object SparseSimd {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = SPECIES.length()

    fun dot(indices: IntArray, values: DoubleArray, y: DoubleArray): Double {
        val len = indices.size
        var k = 0
        val bound = SPECIES.loopBound(len)
        var sum = DoubleVector.zero(SPECIES)
        while (k < bound) {
            // One AVX2 vgatherqpd, where the scalar loop issues a load per entry.
            val gathered = DoubleVector.fromArray(SPECIES, y, 0, indices, k)
            sum = DoubleVector.fromArray(SPECIES, values, k).fma(gathered, sum)
            k += LANE
        }
        var s = sum.reduceLanes(VectorOperators.ADD)
        while (k < len) {
            s += values[k] * y[indices[k]]
            k++
        }
        return s
    }
}
