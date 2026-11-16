@file:Suppress("LongParameterList") // Primitive arithmetic windows avoid per-call descriptors.

package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*

/** Indexed vector arithmetic over stored sparse positions. */
internal interface IndexedKernels : KernelProvider {
    /**
     * `Σ values(k) · dense(indices(k))` over [count] stored entries (Sparse BLAS `usdot`). Repeated and unsorted
     * indices are permitted. Indices are not checked here; [SparseVector] and [SparseMatrix] check theirs once.
     */
    fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = missingKernel(name)

    /** `Σ x(k) · y(k)` over the positions two stored runs share. Both index runs must be strictly increasing. */
    fun dot(
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
    ): Double = missingKernel(name)

    /**
     * `destination(indices(k)) += alpha · values(k)` (Sparse BLAS `usaxpy`). Indices must be strictly increasing,
     * and [values] must not be [destination]. Every stored product is formed, even for a zero [alpha], which is how a
     * sparse traversal keeps a stored zero's product; the [SparseVector] form skips a zero [alpha] as BLAS does.
     */
    fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ): Unit = missingKernel(name)

    /** `destination(indices(k)) = values(k)` (Sparse BLAS `ussc`), with the index and buffer rules of [axpy]. */
    fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ): Unit = missingKernel(name)

    /** `values(k) = source(indices(k))` (Sparse BLAS `usga`). [values] must not be [source]. */
    fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = missingKernel(name)

    /** [gather], then zero each position read in [source] (Sparse BLAS `usgz`). Indices must be strictly increasing. */
    fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = missingKernel(name)
}

private fun missingKernel(name: String): Nothing =
    throw UnsupportedOperationException("This operation is not supplied by $name")
