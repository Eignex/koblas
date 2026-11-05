package com.eignex.koblas.internal.operations.sparse

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.internal.storage.requireIndex
import com.eignex.koblas.internal.storage.requireShape

internal fun KoblasEngine.executeDotColumn(matrix: SparseMatrix, column: Int, dense: DoubleArray): Double {
    requireIndex(column in 0 until matrix.cols) { "column $column is outside [0, ${matrix.cols})" }
    requireShape(matrix.rows == dense.size) { "dotColumn: sizes differ, ${matrix.rows} vs ${dense.size}" }
    val start = matrix.colPointers[column]
    val count = matrix.colPointers[column + 1] - start
    return dot(matrix.rowIndices, start, matrix.values, start, count, dense)
}
