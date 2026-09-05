package com.eignex.koblas.sparse.internal

import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.forEachStored
import com.eignex.koblas.requireIndex
import com.eignex.koblas.requireShape

/**
 * [a] with every column [replacements] names replaced by the vector it maps to, in one pass.
 *
 * The same thing as [com.eignex.koblas.withColumn] applied in turn, for a caller holding several: each of
 * those copies the structure of the whole matrix and validates it again, so a run of them costs the matrix
 * once per replacement where this costs it once. A column no replacement names is copied as it lies, and a
 * stored zero survives on either side, since replacing a column is structural.
 *
 * Each entering vector must have as many entries as [a] has rows, which is what its column will hold.
 */
@OptIn(UnsafeKoblasApi::class)
internal fun replaceColumns(a: F64SparseMatrix, replacements: Map<Int, F64SparseVector>): F64SparseMatrix {
    if (replacements.isEmpty()) return a
    // Scattered once rather than probed per column. `replacements[j]` boxes its key on every lookup above
    // the Integer cache, and both passes below would run one lookup for each of `a.cols` columns to find
    // the handful that are actually replaced.
    val entering = arrayOfNulls<F64SparseVector>(a.cols)
    for ((column, vector) in replacements) {
        requireIndex(column in 0 until a.cols) { "replaceColumns: column $column is outside 0..${a.cols - 1}" }
        requireShape(vector.size == a.rows) {
            "replaceColumns: entering column $column has ${vector.size} entries, expected ${a.rows}"
        }
        entering[column] = vector
    }
    val colPtr = IntArray(a.cols + 1)
    for (j in 0 until a.cols) {
        var entries = 0
        val entering = entering[j]
        if (entering == null) {
            entries = a.colPtr[j + 1] - a.colPtr[j]
        } else {
            entering.forEachStored { _, _ -> entries++ }
        }
        colPtr[j + 1] = colPtr[j] + entries
    }
    val rowIdx = IntArray(colPtr[a.cols])
    val values = DoubleArray(colPtr[a.cols])
    var at = 0
    for (j in 0 until a.cols) {
        val entering = entering[j]
        if (entering == null) {
            a.forEachInColumn(j) { i, v ->
                rowIdx[at] = i
                values[at] = v
                at++
            }
        } else {
            entering.forEachStored { i, v ->
                rowIdx[at] = i
                values[at] = v
                at++
            }
        }
    }
    return F64SparseMatrix.wrap(a.rows, a.cols, colPtr, rowIdx, values)
}

/**
 * A copy of this vector, for a caller holding it past the call it arrived on. The values of a sparse vector
 * stay live for its owner to write, so what is kept has to be a copy of what it held at the time.
 */
internal fun F64SparseVector.snapshot(): F64SparseVector = F64SparseVector(size, copyIndices(), values.copyOf())
