@file:kotlin.jvm.JvmName("SparseBlas")
@file:kotlin.jvm.JvmMultifileClass
@file:Suppress("TooManyFunctions", "LongParameterList", "VariableNaming", "FunctionParameterNaming", "MaxLineLength")

package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.portable.dense.euclideanNorm
import com.eignex.koblas.internal.sparse.*
import com.eignex.koblas.gemm as matrixGemm

/** Fresh dense product using portable CSC traversal and this engine's arithmetic. */
public fun KoblasEngine.gemm(a: SparseMatrix, b: DenseMatrixRegion, scratch: Scratch? = null): DenseMatrix =
    matrixGemm(1.0, a, false, b, false, scratch) as DenseMatrix

/** Fresh CSC product owning its arrays, with stored zeros retained. */
public fun KoblasEngine.gemm(a: SparseMatrix, b: SparseMatrix): SparseMatrix = gemm(1.0, a, false, b, false)

/**
 * Fresh transposed [a], still CSC, which makes this the CSC-to-CSR conversion as well. Explicitly stored
 * zeros survive, since the transpose is structural rather than arithmetic.
 *
 * On the seam rather than beside the other whole-matrix operations because the two directions of [gemv]
 * leave a caller with no reason to materialize a transpose unless it means to hold it.
 */
public fun KoblasEngine.transpose(a: SparseMatrix): SparseMatrix = transposeCsc(a)

/** `A · x`, or `Aᵀ · x` when [transpose], into a fresh result. */
public fun KoblasEngine.gemv(
    a: SparseMatrix,
    x: DoubleArray,
    transpose: Boolean = false,
    scratch: Scratch? = null,
): DoubleArray {
    val size = if (transpose) a.cols else a.rows
    requireGemvOperands(a, transpose, x.size, size)
    val y = DoubleArray(size)
    gemv(1.0, a, x, 0.0, y, transpose, scratch)
    return y
}

/** `xᵀ·y` for a sparse [x] against a dense [y] (Sparse BLAS `usdot`); walks only the stored entries. */
public fun KoblasEngine.dot(x: SparseVector, y: DoubleArray): Double {
    requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
    return this.dot(x.indices, 0, x.values, 0, x.values.size, y)
}

/**
 * Dot product of [column] of [matrix] with [dense], whose length must equal [SparseMatrix.rows].
 * Only stored entries participate, including explicit zeros; absent positions are never evaluated.
 * Reads the current [SparseMatrix.values] on every call and permits that array to be [dense].
 *
 * Built-in implementations reuse the matrix's validated CSC support and call the indexed dot leaf
 * at the column's offsets without copying or rescanning indices. The matrix's structural arrays must remain unchanged, as
 * [SparseMatrix.wrap] requires; this call retains no arrays or column view after returning.
 *
 * @throws IndexOutOfBoundsException if [column] is outside the matrix's columns.
 * @throws IllegalArgumentException if [dense] has the wrong length.
 */
public fun KoblasEngine.dotColumn(matrix: SparseMatrix, column: Int, dense: DoubleArray): Double {
    validateColumnDot(matrix, column, dense)
    val start = matrix.colPointers[column]
    val count = matrix.colPointers[column + 1] - start
    return this.dot(matrix.rowIndices, start, matrix.values, start, count, dense)
}

/**
 * `xᵀ·y` for two sparse vectors, merging their index lists in one pass. Relies on both operands being
 * strictly ascending, which `SparseVector` validates.
 */
public fun KoblasEngine.dot(x: SparseVector, y: SparseVector): Double {
    requireShape(x.size == y.size) { "dot: sizes differ, ${x.size} vs ${y.size}" }
    return this.dot(x.indices, 0, x.values, 0, x.values.size, y.indices, 0, y.values, 0, y.values.size)
}

/** `y += alpha·x` for a sparse [x] into a dense [y] (Sparse BLAS `usaxpy`), touching only x's stored
 *  positions. */
public fun KoblasEngine.axpy(y: DoubleArray, alpha: Double, x: SparseVector) {
    requireShape(x.size == y.size) { "axpy: sizes differ, ${x.size} vs ${y.size}" }
    if (alpha == 0.0) return
    this.axpy(x.indices, 0, x.values, 0, x.values.size, alpha, y)
}

/** Write [x]'s stored entries into [out] at their positions, leaving the rest of [out] alone (Sparse
 *  BLAS `ussc`). Zero-fill [out] first for a plain densification. */
public fun KoblasEngine.scatter(x: SparseVector, out: DoubleArray) {
    requireShape(x.size == out.size) { "scatter: sizes differ, ${x.size} vs ${out.size}" }
    scatter(x.indices, 0, x.values, 0, x.values.size, out)
}

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
public fun KoblasEngine.gather(x: SparseVector, from: DoubleArray) {
    requireShape(x.size == from.size) { "gather: sizes differ, ${x.size} vs ${from.size}" }
    require(x.values !== from) { "values and source must use distinct buffers" }
    this.gather(x.indices, 0, x.values, 0, x.values.size, from)
}

/**
 * [gather], and zero in [from] the positions it read (Sparse BLAS `usgz`), leaving the rest of [from]
 * alone. The pair is one pass rather than two because the caller of a gather-and-zero wants [from]
 * emptied of exactly what it took.
 *
 * [x]'s values and [from] must be distinct arrays, for the reason [gather] gives.
 */
public fun KoblasEngine.gatherZero(x: SparseVector, from: DoubleArray) {
    requireShape(x.size == from.size) { "gatherZero: sizes differ, ${x.size} vs ${from.size}" }
    require(x.values !== from) { "values and source must use distinct buffers" }
    this.gatherZero(x.indices, 0, x.values, 0, x.values.size, from)
}

/** Euclidean norm over the stored entries, rescaled as the dense [euclideanNorm] is. */
public fun KoblasEngine.nrm2(x: SparseVector): Double = this.nrm2(x.values, 0, x.values.size)

/** `Sum |x_i|` over the stored entries. */
public fun KoblasEngine.asum(x: SparseVector): Double = this.asum(x.values, 0, x.values.size)
