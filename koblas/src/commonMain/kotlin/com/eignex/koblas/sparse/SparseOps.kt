@file:kotlin.jvm.JvmName("SparseBlas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.operands.*

/**
 * Dot product of [column] of [matrix] with [dense], whose length must equal [SparseMatrix.rows].
 * Only stored entries participate, including explicit zeros; absent positions are never evaluated.
 * Reads the current [SparseMatrix.values] on every call and permits that array to be [dense].
 *
 * Built-in implementations reuse the matrix's validated CSC support and call the indexed dot leaf
 * at the column's offsets without copying or rescanning indices. The matrix's structural arrays must
 * remain unchanged, as [SparseMatrix.wrap] requires; this call retains no arrays or column view after returning.
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
