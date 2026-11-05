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

/**
 * Fresh transposed [a], still CSC, which makes this the CSC-to-CSR conversion as well. Explicitly stored
 * zeros survive, since the transpose is structural rather than arithmetic.
 *
 * On the seam rather than beside the other whole-matrix operations because the two directions of [gemv]
 * leave a caller with no reason to materialize a transpose unless it means to hold it.
 */
public fun KoblasEngine.transpose(a: SparseMatrix): SparseMatrix = transposeCsc(a)

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
    requireIndex(column in 0 until matrix.cols) { "column $column is outside [0, ${matrix.cols})" }
    requireShape(matrix.rows == dense.size) { "dotColumn: sizes differ, ${matrix.rows} vs ${dense.size}" }
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

/** Euclidean norm over the stored entries, rescaled as the dense [euclideanNorm] is. */
public fun KoblasEngine.nrm2(x: SparseVector): Double = this.nrm2(x.values, 0, x.values.size)

/** `Sum |x_i|` over the stored entries. */
public fun KoblasEngine.asum(x: SparseVector): Double = this.asum(x.values, 0, x.values.size)
