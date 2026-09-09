package com.eignex.koblas.sparse

import com.eignex.koblas.BackendMetadata
import com.eignex.koblas.BackendMetadataProvider
import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.cKernelsAvailable
import com.eignex.koblas.dense.simdAvailable
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/** Sparse SIMD kernels where the Vector API is present, the bundled C kernels otherwise. */
internal actual object PlatformSparseKernels : SparseKernels, BackendMetadataProvider {
    private val selected: SparseKernels = when {
        simdAvailable -> SimdSparseKernels
        cKernelsAvailable -> CSparseKernels
        else -> ReferenceSparseLinearAlgebra
    }

    actual override val name: String get() = selected.name

    override val isPortable: Boolean get() = selected.isPortable

    override val backendMetadata: BackendMetadata
        get() = (selected as? BackendMetadataProvider)?.backendMetadata ?: BackendMetadata()

    actual override fun dot(x: SparseVector, y: DoubleArray): Double = selected.dot(x, y)

    actual override fun dot(x: SparseVector, y: SparseVector): Double = selected.dot(x, y)

    actual override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) = selected.axpy(y, alpha, x)

    actual override fun scatter(x: SparseVector, out: DoubleArray) = selected.scatter(x, out)

    actual override fun gather(x: SparseVector, from: DoubleArray) = selected.gather(x, from)

    actual override fun gatherZero(x: SparseVector, from: DoubleArray) = selected.gatherZero(x, from)

    actual override fun nrm2(x: SparseVector): Double = selected.nrm2(x)

    actual override fun asum(x: SparseVector): Double = selected.asum(x)
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
