package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.VectorRoute
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.internal.sparse.gather
import com.eignex.koblas.internal.sparse.scatter
import com.eignex.koblas.internal.sparse.validateColumnDot

/**
 * Sparse vector and indexed-slice numerical kernels selected by a [com.eignex.koblas.KoblasEngine].
 *
 * The raw overloads operate directly on caller-owned arrays and allocate nothing. Structural policy remains
 * with the caller.
 */
public interface SparseKernels {
    /** Short implementation identifier for diagnostics. */
    public val name: String

    /**
     * Where a call of this [operation] over [count] stored entries would execute.
     *
     * The same decision the call makes, asked before making it. [name] identifies the selection as a whole and
     * cannot answer this: a Vector API selection still runs the scalar loop for a support shorter than its
     * crossover, for an operation it never vectorised, and on a host whose indexed loads or stores it cannot
     * use. A measurement that reads only [name] publishes those as vector timings.
     *
     * Building a route allocates, so it belongs before a timed region and never inside one.
     */
    public fun routeOf(operation: SparseOperation, count: Int): VectorRoute<SparseOperation>

    /** `xᵀ·y` for a sparse [x] against a dense [y] (Sparse BLAS `usdot`); walks only the stored entries. */
    public fun dot(x: SparseVector, y: DoubleArray): Double

    /**
     * Dot product of [column] of [matrix] with [dense], whose length must equal [SparseMatrix.rows].
     * Only stored entries participate, including explicit zeros; absent positions are never evaluated.
     * Reads the current [SparseMatrix.values] on every call and permits that array to be [dense].
     *
     * Built-in implementations reuse the matrix's validated CSC support and call the indexed dot leaf
     * at the column's offsets without copying or rescanning indices. [routeOf] with [SparseOperation.DotDense]
     * and the column's stored count describes that leaf. Custom implementations default to the validated
     * raw [dot] overload. The matrix's structural arrays must remain unchanged, as [SparseMatrix.wrap] requires;
     * this call retains no arrays or column view after returning.
     *
     * @throws IndexOutOfBoundsException if [column] is outside the matrix's columns.
     * @throws IllegalArgumentException if [dense] has the wrong length.
     */
    public fun dotColumn(matrix: SparseMatrix, column: Int, dense: DoubleArray): Double {
        validateColumnDot(matrix, column, dense)
        val start = matrix.colPointers[column]
        val count = matrix.colPointers[column + 1] - start
        return dot(matrix.rowIndices, start, matrix.values, start, count, dense)
    }

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
     *
     * [x]'s values and [from] must be distinct arrays, as they must be for [scatter]. Reading and writing one
     * array through two index orders has no single answer: a scalar loop moves an entry per step while a
     * vector kernel reads a whole block before storing it, so the two would disagree on the overlap.
     */
    public fun gather(x: SparseVector, from: DoubleArray)

    /**
     * [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`), leaving the rest of [from]
     * alone. The pair is one pass rather than two because the caller of a gather-and-zero wants [from]
     * emptied of exactly what it took.
     *
     * [x]'s values and [from] must be distinct arrays, for the reason [gather] gives.
     */
    public fun gatherZero(x: SparseVector, from: DoubleArray)

    /** Euclidean norm over the stored entries, rescaled as the dense [euclideanNorm] is. */
    public fun nrm2(x: SparseVector): Double

    /**
     * Euclidean norm of [values] at the [count] positions in [indices]. Repeated and unsorted indices are
     * permitted and each occurrence contributes once. Finite-range rescaling protects against avoidable
     * square-sum overflow or underflow; unspecified exceptional results follow the selected implementation.
     * The touched window and all selected indices are validated before arithmetic.
     */
    public fun nrm2(indices: IntArray, indexOffset: Int, count: Int, values: DoubleArray): Double

    /** `Sum |x_i|` over the stored entries. */
    public fun asum(x: SparseVector): Double
}
