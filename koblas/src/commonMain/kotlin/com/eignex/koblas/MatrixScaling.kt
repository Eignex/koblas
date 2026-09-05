@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("MatrixOpsKt")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

// Part of the MatrixOpsKt facade. Splitting the file would otherwise rename the class JVM callers
// compiled against, so the four parts are joined back into one rather than becoming four.

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix

/** Scale row `i` by d(i) in place, the product `D * A` for the diagonal D with entries d(i). */
public fun F64DenseMatrix.scaleRows(d: DoubleArray) {
    requireShape(d.size == rows) { "scaleRows: d length ${d.size} != $rows rows" }
    val ad = data
    for (j in 0 until cols) {
        val base = j * rows
        for (i in 0 until rows) ad[base + i] *= d[i]
    }
}

/** Scale column `j` by d(j) in place, the product `A * D` for the diagonal D with entries d(j). */
public fun F64DenseMatrix.scaleColumns(d: DoubleArray) {
    requireShape(d.size == cols) { "scaleColumns: d length ${d.size} != $cols columns" }
    val kernels = koblas.kernels
    for (j in 0 until cols) {
        val f = d[j]
        if (f != 1.0) kernels.scale(data, j * rows, f, rows)
    }
}

/**
 * Zero the strict upper triangle in place, leaving the diagonal and everything below it untouched.
 *
 * A factorization promises nothing above its diagonal: the matrix behind
 * [com.eignex.koblas.dense.F64CholeskyDecomposition.l] holds whatever the backend happened to write there,
 * which is not factorization data and differs from one backend to the next. A factor that gets compared,
 * hashed or serialized has to be cleaned first, so that two mathematically equal factors agree entry for
 * entry. The `lowerFactor` of [com.eignex.koblas.dense.F64CholeskyDecomposition] answers the same need by
 * copying; this keeps the matrix the caller already holds.
 *
 * Each column's strict upper entries are contiguous in column-major storage, so this is one fill per column
 * rather than an indexed walk. In a matrix wider than it is tall, every column past the last row lies
 * entirely above the diagonal and is zeroed whole.
 */
public fun F64DenseMatrix.zeroStrictUpper() {
    for (j in 1 until cols) {
        val start = j * rows
        data.fill(0.0, start, start + minOf(j, rows))
    }
}

/** Scale column `j` by d(j) in place for a CSC matrix. The pattern is untouched. */
public fun F64SparseMatrix.scaleColumns(d: DoubleArray) {
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
 * unchanged. The matrix remains mutable through [F64SparseMatrix.values].
 */
public fun F64SparseMatrix.scaleRows(d: DoubleArray) {
    requireShape(d.size == rows) { "scaleRows: d length ${d.size} != $rows rows" }
    for (k in values.indices) values[k] *= d[rowIdx[k]]
}
