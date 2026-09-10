package com.eignex.koblas.sparse

import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseVectorKernels

/** Numerical leaves over trusted sorted indexed slices. Callers own shape and slice validation. */
internal interface IndexedSparseKernels {
    fun dotDense(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, dense: DoubleArray): Double

    @Suppress("LongParameterList")
    fun dotSparse(
        xIndices: IntArray,
        xValues: DoubleArray,
        xFromIndex: Int,
        xToIndex: Int,
        yIndices: IntArray,
        yValues: DoubleArray,
        yFromIndex: Int,
        yToIndex: Int,
    ): Double

    fun axpy(
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        alpha: Double,
        destination: DoubleArray,
    )

    fun scatter(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, destination: DoubleArray)

    fun gather(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, source: DoubleArray)

    fun gatherZero(indices: IntArray, values: DoubleArray, fromIndex: Int, toIndex: Int, source: DoubleArray)
}

/** Sparse vector and matrix leaves selected together once for an immutable engine. */
internal class SparseKernelFamilies(
    val vector: SparseKernels,
    val indexed: IndexedSparseKernels,
    values: DenseVectorKernels,
    panels: DensePanelKernels,
) {
    val panel = SparsePanelKernels(values, panels)
}

internal expect val platformSparseKernelFamilies: SparseKernelFamilies
