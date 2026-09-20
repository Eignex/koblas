package com.eignex.koblas.sparse

/** JVM Vector API indexed kernels, with scalar fallbacks below measured call widths. */
internal object SimdIndexedSparseKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    private val vectorScatter = configuredJvmVectorScatter()
    private val vectorGather = SparseSimd.autoGatherEligible

    override val name: String = "simd"

    override fun implementationFor(operation: SparseOperation, count: Int): String? = when {
        !usesVector(operation, count) -> ScalarIndexedSparseKernels.name
        operation == SparseOperation.IndexedNrm2 -> null
        else -> name
    }

    /**
     * Whether this call reaches a vector kernel, which is the one decision every override below makes.
     *
     * Both the dispatch and [implementationFor] read it, so the route a benchmark is given and the path the
     * call takes cannot disagree. The two masks differ because the host can support an indexed load without
     * supporting an indexed store.
     */
    private fun usesVector(operation: SparseOperation, count: Int): Boolean {
        if (count < SparseTuning.simdIndexedCrossover || count < SparseSimd.lanes) return false
        return when (operation) {
            SparseOperation.DotDense, SparseOperation.Gather, SparseOperation.IndexedNrm2 -> vectorGather

            SparseOperation.Axpy, SparseOperation.Scatter, SparseOperation.GatherZero -> vectorScatter

            // A sparse-sparse dot has no vector form here, and the whole-vector reductions never reach the
            // indexed kernels at all: they are contiguous runs the dense kernels take.
            SparseOperation.DotSparse, SparseOperation.Nrm2, SparseOperation.Asum -> false
        }
    }

    override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = if (usesVector(SparseOperation.DotDense, count)) {
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
        if (usesVector(SparseOperation.Axpy, count)) {
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
        if (usesVector(SparseOperation.Scatter, count)) {
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
        if (usesVector(SparseOperation.Gather, count)) {
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
        if (usesVector(SparseOperation.GatherZero, count)) {
            SparseSimd.gatherZero(indices, indexOffset, values, valueOffset, count, source)
        } else {
            ScalarIndexedSparseKernels.gatherZero(indices, indexOffset, values, valueOffset, count, source)
        }
    }

    override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double =
        if (usesVector(SparseOperation.IndexedNrm2, count)) {
            SparseSimd.nrm2(indices, indexOffset, count, values)
        } else {
            ScalarIndexedSparseKernels.nrm2(indices, indexOffset, count, values)
        }
}
