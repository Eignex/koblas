package com.eignex.koblas.sparse

import com.eignex.koblas.internal.kernels.JvmCKernelBindings

/** Bundled C indexed kernels, with operations below their measured crossover retained by the scalar delegate. */
internal object CIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    override fun dotDense(indices: IntArray, values: DoubleArray, dense: DoubleArray): Double =
        if (values.size >= SparseTuning.dotDenseCCrossover) {
            JvmCKernelBindings.sparseDotDense(indices, values, dense)
        } else {
            ScalarIndexedSparseKernels.dotDense(indices, values, dense)
        }
}

/** JVM Vector API indexed data movement, with ordered matrix arithmetic retained by the scalar delegate. */
internal object SimdIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    private val vectorScatter = configuredJvmVectorScatter()

    override fun dotDense(indices: IntArray, values: DoubleArray, dense: DoubleArray): Double =
        SparseSimd.dot(indices, values, dense)

    override fun axpy(indices: IntArray, values: DoubleArray, alpha: Double, destination: DoubleArray) {
        if (vectorScatter) {
            SparseSimd.axpy(indices, values, destination, alpha)
        } else {
            ScalarIndexedSparseKernels.axpy(indices, values, alpha, destination)
        }
    }

    override fun scatter(indices: IntArray, values: DoubleArray, destination: DoubleArray) {
        if (vectorScatter) {
            SparseSimd.scatter(indices, values, destination)
        } else {
            ScalarIndexedSparseKernels.scatter(indices, values, destination)
        }
    }

    override fun gather(indices: IntArray, values: DoubleArray, source: DoubleArray) =
        SparseSimd.gather(indices, values, source)

    override fun gatherZero(indices: IntArray, values: DoubleArray, source: DoubleArray) {
        if (vectorScatter) {
            SparseSimd.gatherZero(indices, values, source)
        } else {
            ScalarIndexedSparseKernels.gatherZero(indices, values, source)
        }
    }
}
