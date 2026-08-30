package com.eignex.koblas.sparse

import com.eignex.koblas.BackendMetadataProvider
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.F64CKernels
import com.eignex.koblas.dense.F64SimdKernels
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.requireShape

/** The bundled C sparse kernels without automatic SIMD selection. */
internal object F64CSparseKernels : F64SparseKernels {
    override val name: String get() = BackendNames.C_SPARSE

    override val isPortable: Boolean get() = true

    override val isAvailable: Boolean get() = JvmCKernelBindings.isAvailable

    override fun dot(x: F64SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return JvmCKernelBindings.sparseDotDense(x.indices, x.values, y)
    }

    override fun dot(x: F64SparseVector, y: F64SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return JvmCKernelBindings.sparseDotSparse(x.indices, x.values, y.indices, y.values)
    }

    override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector) {
        requireShape(y.size == x.size) { "axpy: sizes differ, ${y.size} vs ${x.size}" }
        if (alpha != 0.0) JvmCKernelBindings.sparseAxpy(x.indices, x.values, alpha, y)
    }

    override fun scatter(x: F64SparseVector, out: DoubleArray) {
        requireShape(out.size == x.size) { "scatter: sizes differ, ${out.size} vs ${x.size}" }
        JvmCKernelBindings.sparseScatter(x.indices, x.values, out)
    }

    override fun gather(x: F64SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gather: sizes differ, ${from.size} vs ${x.size}" }
        JvmCKernelBindings.sparseGather(x.indices, x.values, from)
    }

    override fun gatherZero(x: F64SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gatherZero: sizes differ, ${from.size} vs ${x.size}" }
        JvmCKernelBindings.sparseGatherZero(x.indices, x.values, from)
    }

    override fun nrm2(x: F64SparseVector): Double = F64CKernels.nrm2(x.values, 0, x.values.size)

    override fun asum(x: F64SparseVector): Double = F64CKernels.asum(x.values, 0, x.values.size)
}

/** The JVM Vector API sparse kernels without automatic C selection. */
internal object F64SimdSparseKernels : F64SparseKernels, BackendMetadataProvider {
    private val scatter = JvmVectorScatter.configured()

    override val name: String get() = BackendNames.SIMD_SPARSE

    override val isPortable: Boolean get() = true

    override val isAvailable: Boolean get() = F64SimdKernels.isAvailable

    override val backendMetadata get() = scatter.metadata

    override fun dot(x: F64SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return SparseSimd.dot(x.indices, x.values, y)
    }

    override fun dot(x: F64SparseVector, y: F64SparseVector): Double = F64ReferenceSparseLinearAlgebra.dot(x, y)

    override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector) {
        requireShape(y.size == x.size) { "axpy: sizes differ, ${y.size} vs ${x.size}" }
        if (alpha == 0.0) return
        if (scatter.enabled) {
            SparseSimd.axpy(x.indices, x.values, y, alpha)
        } else {
            F64ReferenceSparseLinearAlgebra.axpy(y, alpha, x)
        }
    }

    override fun scatter(x: F64SparseVector, out: DoubleArray) {
        requireShape(out.size == x.size) { "scatter: sizes differ, ${out.size} vs ${x.size}" }
        if (scatter.enabled) {
            SparseSimd.scatter(x.indices, x.values, out)
        } else {
            F64ReferenceSparseLinearAlgebra.scatter(x, out)
        }
    }

    override fun gather(x: F64SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gather: sizes differ, ${from.size} vs ${x.size}" }
        SparseSimd.gather(x.indices, x.values, from)
    }

    override fun gatherZero(x: F64SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gatherZero: sizes differ, ${from.size} vs ${x.size}" }
        if (scatter.enabled) {
            SparseSimd.gatherZero(x.indices, x.values, from)
        } else {
            F64ReferenceSparseLinearAlgebra.gatherZero(x, from)
        }
    }

    override fun nrm2(x: F64SparseVector): Double = F64SimdKernels.nrm2(x.values, 0, x.values.size)

    override fun asum(x: F64SparseVector): Double = F64SimdKernels.asum(x.values, 0, x.values.size)
}
