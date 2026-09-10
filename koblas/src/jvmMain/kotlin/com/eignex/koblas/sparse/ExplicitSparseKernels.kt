package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.CKernels
import com.eignex.koblas.dense.CPanelKernels
import com.eignex.koblas.dense.SimdKernels
import com.eignex.koblas.dense.SimdPanelKernels
import com.eignex.koblas.internal.configuration.ImplementationNames
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.requireShape

/** Bundled C indexed leaves with scalar fallbacks for short or interior slices. */
internal object CIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    private val DOT_DENSE_C_CROSSOVER = SparseTuning.dotDenseCCrossover

    val isAvailable: Boolean get() = JvmCKernelBindings.isAvailable

    override fun dotDense(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        dense: DoubleArray,
    ): Double = if (fromIndex == 0 && toIndex == indices.size && toIndex >= DOT_DENSE_C_CROSSOVER) {
        JvmCKernelBindings.sparseDotDense(indices, values, dense)
    } else {
        ScalarIndexedSparseKernels.dotDense(indices, values, fromIndex, toIndex, dense)
    }
}

/** JVM Vector API indexed leaves, retaining scalar order for the interior CSC slices used by matrix algorithms. */
internal object SimdIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    private val scatter = JvmVectorScatter.configured()

    val isAvailable: Boolean get() = SimdKernels.isAvailable

    override fun dotDense(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        dense: DoubleArray,
    ): Double = if (fromIndex == 0 && toIndex == indices.size) {
        SparseSimd.dot(indices, values, dense)
    } else {
        ScalarIndexedSparseKernels.dotDense(indices, values, fromIndex, toIndex, dense)
    }

    override fun axpy(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        alpha: Double,
        destination: DoubleArray,
    ) {
        if (fromIndex == 0 && toIndex == indices.size && scatter.enabled) {
            SparseSimd.axpy(indices, values, destination, alpha)
        } else {
            ScalarIndexedSparseKernels.axpy(indices, values, fromIndex, toIndex, alpha, destination)
        }
    }

    override fun scatter(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        destination: DoubleArray,
    ) {
        if (fromIndex == 0 && toIndex == indices.size && scatter.enabled) {
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
        if (fromIndex == 0 && toIndex == indices.size && scatter.enabled) {
            SparseSimd.gatherZero(indices, values, source)
        } else {
            ScalarIndexedSparseKernels.gatherZero(indices, values, fromIndex, toIndex, source)
        }
    }
}

internal object CSparseKernels : SparseKernels {
    override val name: String get() = ImplementationNames.C_SPARSE

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return CIndexedSparseKernels.dotDense(x.indices, x.values, 0, x.values.size, y)
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
    private val scatter = JvmVectorScatter.configured()

    override val name: String get() = ImplementationNames.SIMD_SPARSE

    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return SparseSimd.dot(x.indices, x.values, y)
    }

    override fun dot(x: SparseVector, y: SparseVector): Double = ScalarSparseKernels.dot(x, y)

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        requireShape(y.size == x.size) { "axpy: sizes differ, ${y.size} vs ${x.size}" }
        if (alpha == 0.0) return
        if (scatter.enabled) SparseSimd.axpy(x.indices, x.values, y, alpha) else ScalarSparseKernels.axpy(y, alpha, x)
    }

    override fun scatter(x: SparseVector, out: DoubleArray) {
        requireShape(out.size == x.size) { "scatter: sizes differ, ${out.size} vs ${x.size}" }
        if (scatter.enabled) SparseSimd.scatter(x.indices, x.values, out) else ScalarSparseKernels.scatter(x, out)
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
            ScalarSparseKernels.gatherZero(x, from)
        }
    }

    override fun nrm2(x: SparseVector): Double = SimdKernels.nrm2(x.values, 0, x.values.size)

    override fun asum(x: SparseVector): Double = SimdKernels.asum(x.values, 0, x.values.size)
}

internal val cSparseKernelFamilies: SparseKernelFamilies = sparseKernelFamilies(
    CSparseKernels,
    CIndexedSparseKernels,
    CKernels,
    CPanelKernels,
)

internal val simdSparseKernelFamilies: SparseKernelFamilies = sparseKernelFamilies(
    SimdSparseKernels,
    SimdIndexedSparseKernels,
    SimdKernels,
    SimdPanelKernels,
)
