package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.CKernels
import com.eignex.koblas.dense.CPanelKernels
import com.eignex.koblas.dense.SimdKernels
import com.eignex.koblas.dense.SimdPanelKernels
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.requireShape

/** JVM Vector API indexed data movement, with ordered matrix arithmetic retained by the scalar delegate. */
internal object SimdIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    private val vectorScatter = configuredJvmVectorScatter()

    override fun scatter(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        destination: DoubleArray,
    ) {
        if (fromIndex == 0 && toIndex == indices.size && vectorScatter) {
            SparseSimd.scatter(indices, values, destination)
        } else {
            ScalarIndexedSparseKernels.scatter(indices, values, fromIndex, toIndex, destination)
        }
    }

    override fun gather(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, source: DoubleArray) {
        if (fromIndex == 0 && toIndex == indices.size) {
            SparseSimd.gather(indices, values, source)
        } else {
            ScalarIndexedSparseKernels.gather(indices, values, fromIndex, toIndex, source)
        }
    }

    override fun gatherZero(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        source: DoubleArray,
    ) {
        if (fromIndex == 0 && toIndex == indices.size && vectorScatter) {
            SparseSimd.gatherZero(indices, values, source)
        } else {
            ScalarIndexedSparseKernels.gatherZero(indices, values, fromIndex, toIndex, source)
        }
    }
}

internal object CSparseKernels : SparseKernels {
    override val name: String get() = "c-sparse"

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return if (x.values.size >= SparseTuning.dotDenseCCrossover) {
            JvmCKernelBindings.sparseDotDense(x.indices, x.values, y)
        } else {
            ScalarIndexedSparseKernels.dotDense(x.indices, x.values, 0, x.values.size, y)
        }
    }

    override fun dot(x: SparseVector, y: SparseVector): Double = ScalarSparseKernels.dot(x, y)

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) = ScalarSparseKernels.axpy(y, alpha, x)

    override fun scatter(x: SparseVector, out: DoubleArray) = ScalarSparseKernels.scatter(x, out)

    override fun gather(x: SparseVector, from: DoubleArray) = ScalarSparseKernels.gather(x, from)

    override fun gatherZero(x: SparseVector, from: DoubleArray) = ScalarSparseKernels.gatherZero(x, from)

    override fun nrm2(x: SparseVector): Double = CKernels.nrm2(x.values, 0, x.values.size)

    override fun asum(x: SparseVector): Double = CKernels.asum(x.values, 0, x.values.size)
}

internal object SimdSparseKernels : SparseKernels {
    private val vectorScatter = configuredJvmVectorScatter()

    override val name: String get() = "simd-sparse"

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return SparseSimd.dot(x.indices, x.values, y)
    }

    override fun dot(x: SparseVector, y: SparseVector): Double = ScalarSparseKernels.dot(x, y)

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        requireShape(y.size == x.size) { "axpy: sizes differ, ${y.size} vs ${x.size}" }
        if (alpha == 0.0) return
        if (vectorScatter) SparseSimd.axpy(x.indices, x.values, y, alpha) else ScalarSparseKernels.axpy(y, alpha, x)
    }

    override fun scatter(x: SparseVector, out: DoubleArray) {
        requireShape(out.size == x.size) { "scatter: sizes differ, ${out.size} vs ${x.size}" }
        if (vectorScatter) SparseSimd.scatter(x.indices, x.values, out) else ScalarSparseKernels.scatter(x, out)
    }

    override fun gather(x: SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gather: sizes differ, ${from.size} vs ${x.size}" }
        SparseSimd.gather(x.indices, x.values, from)
    }

    override fun gatherZero(x: SparseVector, from: DoubleArray) {
        requireShape(from.size == x.size) { "gatherZero: sizes differ, ${from.size} vs ${x.size}" }
        if (vectorScatter) {
            SparseSimd.gatherZero(x.indices, x.values, from)
        } else {
            ScalarSparseKernels.gatherZero(x, from)
        }
    }

    override fun nrm2(x: SparseVector): Double = SimdKernels.nrm2(x.values, 0, x.values.size)

    override fun asum(x: SparseVector): Double = SimdKernels.asum(x.values, 0, x.values.size)
}

internal val cSparseKernelFamilies: SparseKernelFamilies = SparseKernelFamilies(
    CSparseKernels,
    ScalarIndexedSparseKernels,
    CKernels,
    CPanelKernels,
)

internal val simdSparseKernelFamilies: SparseKernelFamilies = SparseKernelFamilies(
    SimdSparseKernels,
    SimdIndexedSparseKernels,
    SimdKernels,
    SimdPanelKernels,
)
