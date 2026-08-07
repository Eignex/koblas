package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.simdAvailable
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/**
 * The nonzero count from which the gathered `usdot` beats the portable loop.
 *
 * 128 because that is the smallest measured size where it won (1.17x, against a 23 per cent loss at 32). The
 * crossover is somewhere between, and 64 is not measured — this is deliberately the conservative end of the
 * interval, so no size regresses. See the threshold task before treating it as tuned.
 */
private const val SPARSE_DOT_SIMD_MIN = 128

/** JVM sparse level-1 kernels: a gathered `usdot` above [SPARSE_DOT_SIMD_MIN], the portable loops below. */
internal actual object PlatformSparseVectorKernels : SparseVectorKernels {
    actual override val name: String get() = if (simdAvailable) "simd-sparse" else "reference"

    /**
     * `usdot` with the dense operand gathered a register at a time.
     *
     * `super.dot` below the threshold rather than a copy of the loop, so the two paths cannot drift: the
     * portable implementation is the interface default and this defers to it.
     */
    actual override fun dot(x: SparseVector, y: DoubleArray): Double {
        require(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        if (!simdAvailable || x.indices.size < SPARSE_DOT_SIMD_MIN) return super.dot(x, y)
        return SparseSimd.dot(x.indices, x.values, y)
    }
}

/**
 * The vector code, in its own object so its initializer — which touches `DoubleVector` — only runs once the
 * incubator module has been confirmed present. Same arrangement as the dense `Simd` object, for the same
 * reason: a JVM started without `--add-modules=jdk.incubator.vector` has to load the enclosing class
 * cleanly and take the scalar path.
 */
private object SparseSimd {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = SPECIES.length()

    fun dot(indices: IntArray, values: DoubleArray, y: DoubleArray): Double {
        val len = indices.size
        var k = 0
        val bound = SPECIES.loopBound(len)
        var sum = DoubleVector.zero(SPECIES)
        while (k < bound) {
            // The gather: one AVX2 vgatherqpd, where the scalar loop issues a load per entry.
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
