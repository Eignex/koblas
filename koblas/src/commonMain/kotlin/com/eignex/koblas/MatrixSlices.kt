@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DenseBlas

/** Column `j` as a fresh vector, copied rather than viewed. */
public fun DenseMatrix.column(j: Int): DenseVector {
    requireIndex(j in 0 until cols) { "column $j outside [0,$cols)" }
    val start = j * rows
    return DenseVector.wrap(data.copyOfRange(start, start + rows))
}

/** Row `i` as a fresh vector, gathered across the backing. Prefer [column] where the algorithm allows. */
public fun DenseMatrix.row(i: Int): DenseVector {
    requireIndex(i in 0 until rows) { "row $i outside [0,$rows)" }
    val out = DoubleArray(cols)
    for (j in 0 until cols) out[j] = data[i + j * rows]
    return DenseVector.wrap(out)
}

/**
 * Column [j] as a fresh sparse vector. Runs in `O(nnzⱼ)` time and allocates copies of its stored indices and
 * values, so later mutations to the returned vector's indices or values cannot affect this matrix. Explicitly
 * stored zeros are preserved.
 */
public fun SparseMatrix.column(j: Int): SparseVector {
    requireIndex(j in 0 until cols) { "column $j outside [0,$cols)" }
    val start = colPtr[j]
    val end = colPtr[j + 1]
    return SparseVector.wrap(rows, rowIdx.copyOfRange(start, end), values.copyOfRange(start, end))
}

/**
 * Row [i] as a fresh sparse vector whose stored positions are the source columns. Runs in `O(nnz)` time, scanning
 * every stored entry in the matrix to gather the ones in this row, and allocates arrays sized to its stored
 * entries. The returned vector is independent of this matrix, and explicitly stored zeros are preserved.
 *
 * Extracting every row this way costs `O(nnz * rows)`; use [transpose] once and read its columns instead when
 * the algorithm needs many rows.
 */
public fun SparseMatrix.row(i: Int): SparseVector {
    requireIndex(i in 0 until rows) { "row $i outside [0,$rows)" }
    var count = 0
    for (j in 0 until cols) {
        for (k in colPtr[j] until colPtr[j + 1]) if (rowIdx[k] == i) count++
    }
    val indices = IntArray(count)
    val out = DoubleArray(count)
    var n = 0
    for (j in 0 until cols) {
        for (k in colPtr[j] until colPtr[j + 1]) {
            if (rowIdx[k] == i) {
                indices[n] = j
                out[n] = values[k]
                n++
            }
        }
    }
    return SparseVector.wrap(cols, indices, out)
}

/**
 * Fresh transposed matrix, with the active backend ([koblas]). For products, prefer the transpose flags on
 * gemv and gemm, which read the original storage without copying. See [DenseBlas.transpose].
 */
public fun DenseMatrix.transpose(): DenseMatrix = koblas.transpose(this)

/**
 * Fresh transposed matrix, still CSC, which makes this the CSC-to-CSR conversion as well, with the active
 * backend ([koblas]).
 */
public fun SparseMatrix.transpose(): SparseMatrix = koblas.transpose(this)

/**
 * Fresh matrix with column [column] replaced by [entering], still CSC. The replacement is structural, so an
 * explicitly stored zero in [entering] survives as one.
 */
public fun SparseMatrix.withColumn(column: Int, entering: SparseVector): SparseMatrix {
    requireIndex(column in 0 until cols) { "withColumn: column $column outside [0,$cols)" }
    requireShape(entering.size == rows) { "withColumn: entering size ${entering.size}, expected $rows" }
    val start = colPtr[column]
    val end = colPtr[column + 1]
    val delta = entering.indices.size - (end - start)
    val pointers = IntArray(cols + 1) { colPtr[it] + if (it <= column) 0 else delta }
    val outIdx = IntArray(rowIdx.size + delta)
    val outVal = DoubleArray(values.size + delta)
    rowIdx.copyInto(outIdx, endIndex = start)
    values.copyInto(outVal, endIndex = start)
    entering.indices.copyInto(outIdx, start)
    entering.values.copyInto(outVal, start)
    rowIdx.copyInto(outIdx, start + entering.indices.size, end)
    values.copyInto(outVal, start + entering.indices.size, end)
    return SparseMatrix(rows, cols, pointers, outIdx, outVal)
}
