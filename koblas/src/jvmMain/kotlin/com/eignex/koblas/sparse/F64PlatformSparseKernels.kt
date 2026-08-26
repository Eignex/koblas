package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.requireShape
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/** A gathered usdot when the Vector API is present, the portable loop otherwise. */
internal actual object F64PlatformSparseKernels : F64SparseKernels {
    actual override val name: String get() = if (simdAvailable) BackendNames.SIMD_SPARSE else BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** Computes usdot with the dense operand gathered a register at a time. */
    actual override fun dot(x: F64SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        if (!simdAvailable) return F64ReferenceSparseLinearAlgebra.dot(x, y)
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

    actual override fun gather(x: F64SparseVector, from: DoubleArray): Unit =
        F64ReferenceSparseLinearAlgebra.gather(x, from)

    actual override fun gatherZero(x: F64SparseVector, from: DoubleArray): Unit =
        F64ReferenceSparseLinearAlgebra.gatherZero(x, from)

    // Both reduce over the stored values alone, so the indices are beside the point and the dense kernels
    // apply unchanged. They branch on lane width themselves, so this needs no `simdAvailable` guard.
    actual override fun nrm2(x: F64SparseVector): Double = F64PlatformKernels.nrm2(x.values, 0, x.values.size)

    actual override fun asum(x: F64SparseVector): Double = F64PlatformKernels.asum(x.values, 0, x.values.size)
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
