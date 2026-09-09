package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import kotlin.math.min

/** Transposes a column-major matrix into another column-major buffer using cache-sized square tiles. */
internal fun transposeBlocked(src: DoubleArray, rows: Int, cols: Int, dst: DoubleArray) {
    var column = 0
    while (column < cols) {
        val columnEnd = min(column + DenseTuning.transposeBlock, cols)
        var row = 0
        while (row < rows) {
            val rowEnd = min(row + DenseTuning.transposeBlock, rows)
            var sourceColumn = column
            while (sourceColumn < columnEnd) {
                val source = sourceColumn * rows
                var sourceRow = row
                while (sourceRow < rowEnd) {
                    dst[sourceColumn + sourceRow * cols] = src[source + sourceRow]
                    sourceRow++
                }
                sourceColumn++
            }
            row = rowEnd
        }
        column = columnEnd
    }
}

/** Gathers each matrix row segment into [row], applies [operation], then writes the segment back. */
internal inline fun forEachRow(
    n: Int,
    b: DenseMatrix,
    row: DoubleArray,
    columnOffset: Int,
    operation: (DoubleArray) -> Unit,
) {
    if (n == 0) return
    val rows = b.rows
    val data = b.data
    repeat(rows) { i ->
        repeat(n) { j -> row[j] = data[i + (columnOffset + j) * rows] }
        operation(row)
        repeat(n) { j -> data[i + (columnOffset + j) * rows] = row[j] }
    }
}
