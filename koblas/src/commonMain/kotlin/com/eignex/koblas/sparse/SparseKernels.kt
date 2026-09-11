package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.requireIndex
import com.eignex.koblas.requireShape

/**
 * Sparse vector and indexed-slice numerical kernels selected by a [com.eignex.koblas.KoblasEngine].
 *
 * The raw overloads operate directly on caller-owned arrays and do not allocate or borrow a
 * [com.eignex.koblas.Workspace]. Structural policy remains with the caller.
 */
public interface SparseKernels {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /** `xᵀ·y` for a sparse [x] against a dense [y] (Sparse BLAS `usdot`); walks only the stored entries. */
    public fun dot(x: SparseVector, y: DoubleArray): Double

    /**
     * Returns the dot product of [count] indexed [values] and [dense]. Index and value windows are independent.
     * Repeated and unsorted indices are permitted: every supplied entry contributes once. A zero [count] permits
     * offsets at the corresponding array ends. Both windows and every selected index are validated before
     * arithmetic. [values] may be [dense] because this operation only reads its inputs.
     */
    @Suppress("LongParameterList")
    public fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double

    /**
     * `xᵀ·y` for two sparse vectors, merging their index lists in one pass. Relies on both operands being
     * strictly ascending, which `SparseVector` validates.
     */
    public fun dot(x: SparseVector, y: SparseVector): Double

    /** `y += alpha·x` for a sparse [x] into a dense [y] (Sparse BLAS `usaxpy`), touching only x's stored
     *  positions. */
    public fun axpy(y: DoubleArray, alpha: Double, x: SparseVector)

    /**
     * Performs `destination[indices(k)] += alpha * values(k)` over the supplied windows.
     *
     * Selected indices must be strictly increasing, making every destination unique. [values] and [destination]
     * must be distinct arrays. All preconditions are validated before mutation. When [alpha] is either signed
     * zero, no value or destination element is read and [destination] is unchanged, following BLAS `axpy`.
     */
    @Suppress("LongParameterList")
    public fun axpy(
        destination: DoubleArray,
        alpha: Double,
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
    )

    /** Write [x]'s stored entries into [out] at their positions, leaving the rest of [out] alone (Sparse
     *  BLAS `ussc`). Zero-fill [out] first for a plain densification. */
    public fun scatter(x: SparseVector, out: DoubleArray)

    /**
     * Assigns [count] indexed [values] into [destination], leaving every other element unchanged.
     * Selected indices must be strictly increasing, so duplicate-destination assignment is not permitted.
     * [values] and [destination] must be distinct arrays. Windows, indices, ordering, and aliasing are validated
     * before mutation; explicit stored values, including either signed zero, are assigned unchanged.
     */
    @Suppress("LongParameterList")
    public fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    )

    /**
     * Read [from] at [x]'s stored positions into [x]'s values (Sparse BLAS `usga`), the inverse of
     * [scatter]. The pattern is what [x] already stores and does not change; a position of [from] that is
     * nonzero and unstored stays unread, so this narrows a dense vector to a pattern rather than sparsifying
     * it.
     */
    public fun gather(x: SparseVector, from: DoubleArray)

    /**
     * [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`), leaving the rest of [from]
     * alone. The pair is one pass rather than two because the caller of a gather-and-zero wants [from]
     * emptied of exactly what it took.
     */
    public fun gatherZero(x: SparseVector, from: DoubleArray)

    /** Euclidean norm over the stored entries, rescaled as the dense [euclideanNorm] is. */
    public fun nrm2(x: SparseVector): Double

    /**
     * Euclidean norm of [values] at the [count] positions in [indices]. Repeated and unsorted indices are
     * permitted and each occurrence contributes once. The calculation rescales when a naive square sum leaves
     * the finite `Double` range, propagates NaN, and returns positive infinity for infinite input without NaN.
     * The touched window and all selected indices are validated before arithmetic.
     */
    public fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double

    /** `Sum |x_i|` over the stored entries. */
    public fun asum(x: SparseVector): Double
}

internal class SparseKernelAdapter(
    override val name: String,
    private val denseVectorKernels: DenseVectorKernels,
    private val indexedSparseKernels: IndexedSparseKernels,
) : SparseKernels {
    override fun dot(x: SparseVector, y: DoubleArray): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return indexedSparseKernels.dotDense(x.indices, x.values, y)
    }

    override fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double {
        requireSparseWindow(indices.size, indexOffset, count, "indices")
        requireSparseWindow(values.size, valueOffset, count, "values")
        validateSparseIndices(indices, indexOffset, count, dense.size, strictlyIncreasing = false)
        return indexedSparseKernels.dotDense(indices, indexOffset, values, valueOffset, count, dense)
    }

    override fun dot(x: SparseVector, y: SparseVector): Double {
        requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
        return indexedSparseKernels.dotSparse(x.indices, x.values, y.indices, y.values)
    }

    override fun axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
        requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
        if (alpha == 0.0) return
        indexedSparseKernels.axpy(x.indices, x.values, alpha, y)
    }

    override fun axpy(
        destination: DoubleArray,
        alpha: Double,
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
    ) {
        validateMutation(indices, indexOffset, values, valueOffset, count, destination)
        if (alpha == 0.0) return
        indexedSparseKernels.axpy(indices, indexOffset, values, valueOffset, count, alpha, destination)
    }

    override fun scatter(x: SparseVector, out: DoubleArray) {
        requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
        indexedSparseKernels.scatter(x.indices, x.values, out)
    }

    override fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        validateMutation(indices, indexOffset, values, valueOffset, count, destination)
        indexedSparseKernels.scatter(indices, indexOffset, values, valueOffset, count, destination)
    }

    override fun gather(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gather: sizes differ, ${x.size} vs ${from.size}" }
        indexedSparseKernels.gather(x.indices, x.values, from)
    }

    override fun gatherZero(x: SparseVector, from: DoubleArray) {
        requireShape(x.size == from.size) { "gatherZero: sizes differ, ${x.size} vs ${from.size}" }
        indexedSparseKernels.gatherZero(x.indices, x.values, from)
    }

    override fun nrm2(x: SparseVector): Double = denseVectorKernels.nrm2(x.values, 0, x.values.size)

    override fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double {
        requireSparseWindow(indices.size, indexOffset, count, "indices")
        validateSparseIndices(indices, indexOffset, count, values.size, strictlyIncreasing = false)
        return indexedSparseKernels.nrm2(indices, indexOffset, count, values)
    }

    override fun asum(x: SparseVector): Double = denseVectorKernels.asum(x.values, 0, x.values.size)

    private fun validateMutation(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ) {
        requireSparseWindow(indices.size, indexOffset, count, "indices")
        requireSparseWindow(values.size, valueOffset, count, "values")
        require(values !== destination) { "values and destination must use distinct buffers" }
        validateSparseIndices(indices, indexOffset, count, destination.size, strictlyIncreasing = true)
    }
}

private fun requireSparseWindow(length: Int, offset: Int, count: Int, name: String) {
    require(offset >= 0 && count >= 0 && offset.toLong() + count <= length) {
        "$name window [$offset, ${offset.toLong() + count}) exceeds length $length"
    }
}

private fun validateSparseIndices(
    indices: IntArray,
    offset: Int,
    count: Int,
    dimension: Int,
    strictlyIncreasing: Boolean,
) {
    var previous = -1
    for (k in 0 until count) {
        val index = indices[offset + k]
        requireIndex(index in 0 until dimension) { "index $index is outside [0, $dimension)" }
        if (strictlyIncreasing) {
            require(index > previous) { "indices must be strictly increasing" }
            previous = index
        }
    }
}
