package com.eignex.koblas.sparse

import com.eignex.koblas.F64SparseVector
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireShape
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/** The nonzero count from which the gathered usdot beats the portable loop. */
private const val SPARSE_DOT_SIMD_MIN = 128

/** A gathered usdot above [SPARSE_DOT_SIMD_MIN], the portable loops below. */
internal actual object F64PlatformSparseVectorKernels : F64SparseVectorKernels {
    actual override val name: String get() = if (simdAvailable) BackendNames.SIMD_SPARSE else BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** Computes usdot with the dense operand gathered a register at a time. */
    actual override fun dot(x: F64SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        if (!simdAvailable || x.indices.size < SPARSE_DOT_SIMD_MIN) return F64ReferenceSparseLinearAlgebra.dot(x, y)
        return SparseSimd.dot(x.indices, x.values, y)
    }

    // The routines this backend does not provide run the portable versions. Forwarded explicitly rather than
    // by class delegation, which would route a caller's convenience overloads to the portable routine instead
    // of this one, since a delegated member calls back into the delegate.
    actual override fun dot(x: F64SparseVector, y: F64SparseVector): Double = F64ReferenceSparseLinearAlgebra.dot(x, y)

    actual override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector): Unit =
        F64ReferenceSparseLinearAlgebra.axpy(y, alpha, x)

    actual override fun scatter(x: F64SparseVector, out: DoubleArray): Unit =
        F64ReferenceSparseLinearAlgebra.scatter(x, out)

    actual override fun nrm2(x: F64SparseVector): Double = F64ReferenceSparseLinearAlgebra.nrm2(x)

    actual override fun asum(x: F64SparseVector): Double = F64ReferenceSparseLinearAlgebra.asum(x)
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
