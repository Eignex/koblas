package com.eignex.koblas.sparse

import com.eignex.koblas.BackendMetadataProvider
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.F64PlatformKernels
import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.requireShape
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/** Sparse SIMD kernels where the Vector API is present, the bundled C kernels otherwise. */
internal actual object F64PlatformSparseKernels : F64SparseKernels, BackendMetadataProvider {
    actual override val name: String get() = if (simdAvailable) BackendNames.SIMD_SPARSE else BackendNames.C_SPARSE

    override val isPortable: Boolean get() = true

    private val scatter = JvmVectorScatter.configured()

    override val backendMetadata get() = scatter.metadata

    /** Computes usdot with the dense operand gathered a register at a time. */
    actual override fun dot(x: F64SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        if (!simdAvailable) return JvmCKernelBindings.sparseDotDense(x.indices, x.values, y)
        return SparseSimd.dot(x.indices, x.values, y)
    }

    // Sparse-sparse dot keeps the portable merge. Forwarded explicitly rather than by class delegation, which
    // would route a caller's convenience overloads to the portable routine instead of this one, since a
    // delegated member calls back into the delegate.
    actual override fun dot(x: F64SparseVector, y: F64SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return if (simdAvailable) {
            F64ReferenceSparseLinearAlgebra.dot(x, y)
        } else {
            JvmCKernelBindings.sparseDotSparse(x.indices, x.values, y.indices, y.values)
        }
    }

    actual override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector) {
        requireShape(y.size == x.size) { "axpy: sizes differ, ${y.size} vs ${x.size}" }
        if (alpha == 0.0) return
        if (scatter.enabled) {
            SparseSimd.axpy(x.indices, x.values, y, alpha)
        } else if (simdAvailable) {
            F64ReferenceSparseLinearAlgebra.axpy(y, alpha, x)
        } else {
            JvmCKernelBindings.sparseAxpy(x.indices, x.values, alpha, y)
        }
    }

    actual override fun scatter(x: F64SparseVector, out: DoubleArray) {
        requireShape(out.size == x.size) { "scatter: sizes differ, ${out.size} vs ${x.size}" }
        if (scatter.enabled) {
            SparseSimd.scatter(x.indices, x.values, out)
        } else if (simdAvailable) {
            F64ReferenceSparseLinearAlgebra.scatter(x, out)
        } else {
            JvmCKernelBindings.sparseScatter(x.indices, x.values, out)
        }
    }

    actual override fun gather(x: F64SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gather: sizes differ, ${from.size} vs ${x.size}" }
        if (simdAvailable) {
            SparseSimd.gather(x.indices, x.values, from)
        } else {
            JvmCKernelBindings.sparseGather(x.indices, x.values, from)
        }
    }

    actual override fun gatherZero(x: F64SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gatherZero: sizes differ, ${from.size} vs ${x.size}" }
        if (scatter.enabled) {
            SparseSimd.gatherZero(x.indices, x.values, from)
        } else if (simdAvailable) {
            F64ReferenceSparseLinearAlgebra.gatherZero(x, from)
        } else {
            JvmCKernelBindings.sparseGatherZero(x.indices, x.values, from)
        }
    }

    // Both reduce over the stored values alone, so the indices are beside the point and the dense kernels
    // apply unchanged. They branch on lane width themselves, so this needs no `simdAvailable` guard.
    actual override fun nrm2(x: F64SparseVector): Double = F64PlatformKernels.nrm2(x.values, 0, x.values.size)

    actual override fun asum(x: F64SparseVector): Double = F64PlatformKernels.asum(x.values, 0, x.values.size)
}

/** Its own object so the initializer, which touches DoubleVector, runs only once the module is present. */
internal object SparseSimd {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = SPECIES.length()

    val autoScatterEligible: Boolean
        get() = SPECIES.vectorBitSize() == 512 && System.getProperty("os.arch").orEmpty() in X86_ARCHITECTURES

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

    fun axpy(indices: IntArray, values: DoubleArray, y: DoubleArray, alpha: Double) {
        val len = indices.size
        var k = 0
        val bound = SPECIES.loopBound(len)
        val multiplier = DoubleVector.broadcast(SPECIES, alpha)
        while (k < bound) {
            val old = DoubleVector.fromArray(SPECIES, y, 0, indices, k)
            val increment = DoubleVector.fromArray(SPECIES, values, k)
            increment.fma(multiplier, old).intoArray(y, 0, indices, k)
            k += LANE
        }
        while (k < len) {
            y[indices[k]] += alpha * values[k]
            k++
        }
    }

    fun scatter(indices: IntArray, values: DoubleArray, out: DoubleArray) {
        val len = indices.size
        var k = 0
        val bound = SPECIES.loopBound(len)
        while (k < bound) {
            DoubleVector.fromArray(SPECIES, values, k).intoArray(out, 0, indices, k)
            k += LANE
        }
        while (k < len) {
            out[indices[k]] = values[k]
            k++
        }
    }

    fun gather(indices: IntArray, values: DoubleArray, from: DoubleArray) {
        val len = indices.size
        var k = 0
        val bound = SPECIES.loopBound(len)
        while (k < bound) {
            DoubleVector.fromArray(SPECIES, from, 0, indices, k).intoArray(values, k)
            k += LANE
        }
        while (k < len) {
            values[k] = from[indices[k]]
            k++
        }
    }

    fun gatherZero(indices: IntArray, values: DoubleArray, from: DoubleArray) {
        val len = indices.size
        var k = 0
        val bound = SPECIES.loopBound(len)
        val zero = DoubleVector.zero(SPECIES)
        while (k < bound) {
            DoubleVector.fromArray(SPECIES, from, 0, indices, k).intoArray(values, k)
            zero.intoArray(from, 0, indices, k)
            k += LANE
        }
        while (k < len) {
            values[k] = from[indices[k]]
            from[indices[k]] = 0.0
            k++
        }
    }

    private val X86_ARCHITECTURES = setOf("amd64", "x86_64", "x64")
}
