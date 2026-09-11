@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("MatrixOpsKt")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

/** Scale row `i` by `d(i)` in place for dense or sparse storage: the product `D * A`. */
public fun MatrixStorage.scaleRows(d: DoubleArray) {
    requireShape(d.size == rows) { "scaleRows: d length ${d.size} != $rows rows" }
    when (this) {
        is DenseMatrix -> for (j in 0 until cols) {
            val base = j * rows
            for (i in 0 until rows) data[base + i] *= d[i]
        }

        is SparseMatrix -> for (k in values.indices) values[k] *= d[rowIdx[k]]
    }
}

/** Scale column `j` by `d(j)` in place for dense or sparse storage: the product `A * D`. */
public fun MatrixStorage.scaleColumns(d: DoubleArray) {
    requireShape(d.size == cols) { "scaleColumns: d length ${d.size} != $cols columns" }
    when (this) {
        is DenseMatrix -> for (j in 0 until cols) {
            val factor = d[j]
            if (factor != 1.0) koblas.vectorKernels.scale(data, j * rows, factor, rows)
        }

        is SparseMatrix -> for (j in 0 until cols) {
            val factor = d[j]
            if (factor == 1.0) continue
            for (k in colPtr[j] until colPtr[j + 1]) values[k] *= factor
        }
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
