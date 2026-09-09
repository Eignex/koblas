package com.eignex.koblas.sparse

import com.eignex.koblas.BackendMetadataProvider
import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.CKernels
import com.eignex.koblas.dense.SimdKernels
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.requireShape

/**
 * The bundled C sparse kernels without automatic SIMD selection.
 *
 * Only the dot against a dense vector reaches the library. Measured on `SparseLevel1Benchmark` against the
 * portable kernels, the other five lose at every density and length measured, from four stored entries to
 * six thousand, and the sparse-against-sparse dot loses worst at 0.2x to 0.5x. Both sides of that one run
 * the same merge loop, so what separates them is the crossing itself: four heap arrays are pinned for a
 * call whose body is a handful of branches. Widening the C to AVX2 improved these but did not turn any of
 * them around, which follows from the indexing being indirect where the work is a scattered load rather
 * than a wide one.
 *
 * The names stay honest by measurement rather than by origin: this is the C provider, and the routines it
 * does not route into C are the ones where crossing was measured to cost more than it returns.
 */
internal object CSparseKernels : SparseKernels {
    /**
     * Stored entries from which the dot against a dense vector repays its foreign call. The measurement
     * behind the value is on [SparseTuning.dotDenseCCrossover]; it is bound to a field of this object so
     * the comparison does not reach through the tuning object on every call.
     */
    private val DOT_DENSE_C_CROSSOVER = SparseTuning.dotDenseCCrossover

    override val name: String get() = BackendNames.C_SPARSE

    override val isPortable: Boolean get() = true

    override val isAvailable: Boolean get() = JvmCKernelBindings.isAvailable

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return if (x.indices.size < DOT_DENSE_C_CROSSOVER) {
            ReferenceSparseLinearAlgebra.dot(x, y)
        } else {
            JvmCKernelBindings.sparseDotDense(x.indices, x.values, y)
        }
    }

    override fun dot(x: SparseVector, y: SparseVector): Double = ReferenceSparseLinearAlgebra.dot(x, y)

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) = ReferenceSparseLinearAlgebra.axpy(y, alpha, x)

    override fun scatter(x: SparseVector, out: DoubleArray) = ReferenceSparseLinearAlgebra.scatter(x, out)

    override fun gather(x: SparseVector, from: DoubleArray) = ReferenceSparseLinearAlgebra.gather(x, from)

    override fun gatherZero(x: SparseVector, from: DoubleArray) = ReferenceSparseLinearAlgebra.gatherZero(x, from)

    override fun nrm2(x: SparseVector): Double = CKernels.nrm2(x.values, 0, x.values.size)

    override fun asum(x: SparseVector): Double = CKernels.asum(x.values, 0, x.values.size)
}

/** The JVM Vector API sparse kernels without automatic C selection. */
internal object SimdSparseKernels : SparseKernels, BackendMetadataProvider {
    private val scatter = JvmVectorScatter.configured()

    override val name: String get() = BackendNames.SIMD_SPARSE

    override val isPortable: Boolean get() = true

    override val isAvailable: Boolean get() = SimdKernels.isAvailable

    override val backendMetadata get() = scatter.metadata

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return SparseSimd.dot(x.indices, x.values, y)
    }

    override fun dot(x: SparseVector, y: SparseVector): Double = ReferenceSparseLinearAlgebra.dot(x, y)

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        requireShape(y.size == x.size) { "axpy: sizes differ, ${y.size} vs ${x.size}" }
        if (alpha == 0.0) return
        if (scatter.enabled) {
            SparseSimd.axpy(x.indices, x.values, y, alpha)
        } else {
            ReferenceSparseLinearAlgebra.axpy(y, alpha, x)
        }
    }

    override fun scatter(x: SparseVector, out: DoubleArray) {
        requireShape(out.size == x.size) { "scatter: sizes differ, ${out.size} vs ${x.size}" }
        if (scatter.enabled) {
            SparseSimd.scatter(x.indices, x.values, out)
        } else {
            ReferenceSparseLinearAlgebra.scatter(x, out)
        }
    }

    override fun gather(x: SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gather: sizes differ, ${from.size} vs ${x.size}" }
        SparseSimd.gather(x.indices, x.values, from)
    }

    override fun gatherZero(x: SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gatherZero: sizes differ, ${from.size} vs ${x.size}" }
        if (scatter.enabled) {
            SparseSimd.gatherZero(x.indices, x.values, from)
        } else {
            ReferenceSparseLinearAlgebra.gatherZero(x, from)
        }
    }

    override fun nrm2(x: SparseVector): Double = SimdKernels.nrm2(x.values, 0, x.values.size)

    override fun asum(x: SparseVector): Double = SimdKernels.asum(x.values, 0, x.values.size)
}
