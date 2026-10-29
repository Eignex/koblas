package com.eignex.koblas.internal.sparse

import com.eignex.koblas.*
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.dense.*
import com.eignex.koblas.forEachStored
import com.eignex.koblas.requireIndex
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

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
internal fun replaceColumns(a: SparseMatrix, replacements: Map<Int, SparseVector>): SparseMatrix {
    if (replacements.isEmpty()) return a
    // Index replacements once to avoid boxed map lookups for every column in both passes.
    val byColumn = arrayOfNulls<SparseVector>(a.cols)
    for ((column, vector) in replacements) {
        requireIndex(column in 0 until a.cols) { "replaceColumns: column $column is outside 0..${a.cols - 1}" }
        requireShape(vector.size == a.rows) {
            "replaceColumns: entering column $column has ${vector.size} entries, expected ${a.rows}"
        }
        byColumn[column] = vector
    }
    val colPointers = IntArray(a.cols + 1)
    for (j in 0 until a.cols) {
        var entries = 0
        val entering = byColumn[j]
        if (entering == null) {
            entries = a.colPointers[j + 1] - a.colPointers[j]
        } else {
            entering.forEachStored { _, _ -> entries++ }
        }
        colPointers[j + 1] = colPointers[j] + entries
    }
    val rowIndices = IntArray(colPointers[a.cols])
    val values = DoubleArray(colPointers[a.cols])
    var at = 0
    for (j in 0 until a.cols) {
        val entering = byColumn[j]
        if (entering == null) {
            a.forEachInColumn(j) { i, v ->
                rowIndices[at] = i
                values[at] = v
                at++
            }
        } else {
            entering.forEachStored { i, v ->
                rowIndices[at] = i
                values[at] = v
                at++
            }
        }
    }
    return SparseMatrix.wrap(a.rows, a.cols, colPointers, rowIndices, values)
}

/**
 * A copy of this vector, for a caller holding it past the call it arrived on. The values of a sparse vector
 * stay live for its owner to write, so what is kept has to be a copy of what it held at the time.
 */
internal fun SparseVector.snapshot(): SparseVector = SparseVector(size, copyIndices(), values.copyOf())

/** The stored support a column contributes to a matrix schedule. */
internal enum class SparseColumnShape { Whole, Selected, Strict }

/** Pure column/support enumeration shared by execution and diagnostics, without coefficient reads. */
@OptIn(com.eignex.koblas.UnsafeKoblasApi::class)
internal inline fun forEachSparseColumn(
    a: com.eignex.koblas.SparseMatrix,
    shape: SparseColumnShape = SparseColumnShape.Whole,
    lower: Boolean = true,
    forward: Boolean = true,
    action: (column: Int, from: Int, to: Int) -> Unit,
) {
    forEachTriangleColumn(a.cols, forward) { column ->
        val from = a.colPointers[column]
        val to = a.colPointers[column + 1]
        val first = when (shape) {
            SparseColumnShape.Whole -> from

            SparseColumnShape.Selected -> com.eignex.koblas.internal.sparse.selectedRunStart(
                a.rowIndices,
                from,
                to,
                column,
                lower,
            )

            SparseColumnShape.Strict -> com.eignex.koblas.internal.sparse.strictRunStart(
                a.rowIndices,
                from,
                to,
                column,
                lower,
            )
        }
        val end = when (shape) {
            SparseColumnShape.Whole -> to

            SparseColumnShape.Selected -> com.eignex.koblas.internal.sparse.selectedRunEnd(
                a.rowIndices,
                from,
                to,
                column,
                lower,
            )

            SparseColumnShape.Strict -> com.eignex.koblas.internal.sparse.strictRunEnd(
                a.rowIndices,
                from,
                to,
                column,
                lower,
            )
        }
        action(column, first, end)
    }
}
