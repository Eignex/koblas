package com.eignex.koblas.sparse

/**
 * Routes and execution resolve the same callable leaf without allocating a per-call plan.
 *
 * Arithmetic forwarding is final so a backend adds policy by selecting leaves, rather than maintaining a
 * second set of branches or inheriting arithmetic through Kotlin interface delegation.
 */
internal abstract class DispatchingIndexedSparseKernels : IndexedSparseKernels {
    protected abstract fun select(operation: SparseOperation, count: Int): IndexedSparseKernels

    final override fun implementationFor(operation: SparseOperation, count: Int): String? =
        select(operation, count).implementationFor(operation, count)

    @Suppress("LongParameterList")
    final override fun dotDense(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double =
        select(SparseOperation.DotDense, count).dotDense(indices, indexOffset, values, valueOffset, count, dense)

    @Suppress("LongParameterList")
    final override fun dotSparse(
        xIndices: IntArray,
        xIndexOffset: Int,
        xValues: DoubleArray,
        xValueOffset: Int,
        xCount: Int,
        yIndices: IntArray,
        yIndexOffset: Int,
        yValues: DoubleArray,
        yValueOffset: Int,
        yCount: Int,
    ): Double = select(SparseOperation.DotSparse, minOf(xCount, yCount)).dotSparse(
        xIndices, xIndexOffset, xValues, xValueOffset, xCount,
        yIndices, yIndexOffset, yValues, yValueOffset, yCount,
    )

    @Suppress("LongParameterList")
    final override fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ) = select(SparseOperation.Axpy, count).axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)

    @Suppress("LongParameterList")
    final override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) = select(SparseOperation.Scatter, count).scatter(indices, indexOffset, values, valueOffset, count, destination)

    @Suppress("LongParameterList")
    final override fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) = select(SparseOperation.Gather, count).gather(indices, indexOffset, values, valueOffset, count, source)

    @Suppress("LongParameterList")
    final override fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ) = select(SparseOperation.GatherZero, count).gatherZero(indices, indexOffset, values, valueOffset, count, source)

    final override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double =
        select(SparseOperation.IndexedNrm2, count).nrm2(indices, indexOffset, count, values)
}
