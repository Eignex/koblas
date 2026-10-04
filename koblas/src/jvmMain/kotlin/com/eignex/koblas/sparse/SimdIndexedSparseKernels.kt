package com.eignex.koblas.sparse

/** JVM Vector API indexed kernels, with scalar fallbacks below measured call widths. */
internal object SimdIndexedSparseKernels : DispatchingIndexedSparseKernels() {
    private val vectorScatter = configuredJvmVectorScatter()
    private val vectorGather = SparseSimd.autoGatherEligible

    override val name: String = "simd"

    override fun select(operation: SparseOperation, count: Int): IndexedSparseKernels =
        if (usesVector(operation, count)) SparseSimd else ScalarIndexedSparseKernels

    /**
     * Whether the selected leaf can reach a vector body. The two masks differ because the host can support
     * an indexed load without supporting an indexed store.
     *
     * The lane width bounds the configured crossover rather than being covered by it. A deployment may set
     * the crossover to one, and a support narrower than a lane block leaves every entry to the scalar tail
     * of the vector kernel, which is scalar work whatever entered it.
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
}
