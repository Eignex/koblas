package com.eignex.koblas.sparse

import com.eignex.koblas.internal.kernels.JvmCKernelBindings

/** Bundled C indexed kernels, with operations below their measured crossover retained by the scalar delegate. */
internal object CIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = if (count >= SparseTuning.dotDenseCCrossover) {
        JvmCKernelBindings.sparseDotDense(indices, indexOffset, values, valueOffset, count, dense)
    } else {
        ScalarIndexedSparseKernels.dotDense(indices, indexOffset, values, valueOffset, count, dense)
    }

    override fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ) {
        if (count >= SparseTuning.cIndexedMutationCrossover) {
            JvmCKernelBindings.sparseAxpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
        } else {
            ScalarIndexedSparseKernels.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
        }
    }

    override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        if (count >= SparseTuning.cIndexedMutationCrossover) {
            JvmCKernelBindings.sparseScatter(indices, indexOffset, values, valueOffset, count, destination)
        } else {
            ScalarIndexedSparseKernels.scatter(indices, indexOffset, values, valueOffset, count, destination)
        }
    }

    override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double =
        if (count >= SparseTuning.cIndexedNormCrossover) {
            JvmCKernelBindings.sparseNrm2(indices, indexOffset, count, values)
        } else {
            ScalarIndexedSparseKernels.nrm2(indices, indexOffset, count, values)
        }
}

/** JVM Vector API indexed kernels, with scalar fallbacks below measured call widths. */
internal object SimdIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    private val vectorScatter = configuredJvmVectorScatter()

    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = if (SparseSimd.autoIndexedLoadEligible && count >= SparseTuning.simdIndexedCrossover) {
        SparseSimd.dot(indices, indexOffset, values, valueOffset, count, dense)
    } else {
        ScalarIndexedSparseKernels.dotDense(indices, indexOffset, values, valueOffset, count, dense)
    }

    override fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ) {
        if (vectorScatter && count >= SparseTuning.simdIndexedCrossover) {
            SparseSimd.axpy(indices, indexOffset, values, valueOffset, count, destination, alpha)
        } else {
            ScalarIndexedSparseKernels.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
        }
    }

    override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        if (vectorScatter && count >= SparseTuning.simdIndexedCrossover) {
            SparseSimd.scatter(indices, indexOffset, values, valueOffset, count, destination)
        } else {
            ScalarIndexedSparseKernels.scatter(indices, indexOffset, values, valueOffset, count, destination)
        }
    }

    override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        if (SparseSimd.autoIndexedLoadEligible && count >= SparseTuning.simdIndexedCrossover) {
            SparseSimd.gather(indices, indexOffset, values, valueOffset, count, source)
        } else {
            ScalarIndexedSparseKernels.gather(indices, indexOffset, values, valueOffset, count, source)
        }
    }

    override fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) {
        if (vectorScatter && count >= SparseTuning.simdIndexedCrossover) {
            SparseSimd.gatherZero(indices, indexOffset, values, valueOffset, count, source)
        } else {
            ScalarIndexedSparseKernels.gatherZero(indices, indexOffset, values, valueOffset, count, source)
        }
    }

    override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double =
        if (SparseSimd.autoIndexedLoadEligible && count >= SparseTuning.simdIndexedCrossover) {
            SparseSimd.nrm2(indices, indexOffset, count, values)
        } else {
            ScalarIndexedSparseKernels.nrm2(indices, indexOffset, count, values)
        }
}
