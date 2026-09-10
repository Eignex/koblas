@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("MatrixOpsKt")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix

/** Scale row `i` by d(i) in place, the product `D * A` for the diagonal D with entries d(i). */
public fun DenseMatrix.scaleRows(d: DoubleArray) {
    requireShape(d.size == rows) { "scaleRows: d length ${d.size} != $rows rows" }
    val ad = data
    for (j in 0 until cols) {
        val base = j * rows
        for (i in 0 until rows) ad[base + i] *= d[i]
    }
}

/** Scale column `j` by d(j) in place, the product `A * D` for the diagonal D with entries d(j). */
public fun DenseMatrix.scaleColumns(d: DoubleArray) {
    requireShape(d.size == cols) { "scaleColumns: d length ${d.size} != $cols columns" }
    val kernels = koblas.denseKernelFamilies.vector
    for (j in 0 until cols) {
        val f = d[j]
        if (f != 1.0) kernels.scale(data, j * rows, f, rows)
    }
}

/**
 * Zero the strict upper triangle in place, leaving the diagonal and everything below it untouched.
 *
 * This is useful when only the lower triangle of a symmetric matrix should remain authoritative before the
 * matrix is compared, hashed, or serialized.
 *
 * Each column's strict upper entries are contiguous in column-major storage, so this is one fill per column
 * rather than an indexed walk. In a matrix wider than it is tall, every column past the last row lies
 * entirely above the diagonal and is zeroed whole.
 */
public fun DenseMatrix.zeroStrictUpper() {
    for (j in 1 until cols) {
        val start = j * rows
        data.fill(0.0, start, start + minOf(j, rows))
    }
}

/** Scale column `j` by d(j) in place for a CSC matrix. The pattern is untouched. */
public fun SparseMatrix.scaleColumns(d: DoubleArray) {
    requireShape(d.size == cols) { "scaleColumns: d length ${d.size} != $cols columns" }
    for (j in 0 until cols) {
        val f = d[j]
        if (f == 1.0) continue
        for (k in colPtr[j] until colPtr[j + 1]) values[k] *= f
    }
}

/**
 * Scales row `i` by d(i) in place, the product `D * A` for the diagonal D with entries d(i).
 *
 * Runs in `O(nnz)` time, allocates nothing, and keeps the CSC pattern, including explicitly stored zeros,
 * unchanged. The matrix remains mutable through [SparseMatrix.values].
 */
public fun SparseMatrix.scaleRows(d: DoubleArray) {
    requireShape(d.size == rows) { "scaleRows: d length ${d.size} != $rows rows" }
    for (k in values.indices) values[k] *= d[rowIdx[k]]
}
