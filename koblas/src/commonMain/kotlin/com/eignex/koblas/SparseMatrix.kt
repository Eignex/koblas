package com.eignex.koblas

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Compressed-sparse-column form: column j occupies colPtr(j) until colPtr(j + 1) of [rowIdx] and [values],
 * rows strictly ascending. A stored zero is preserved, and a 64-bit-index host library needs a widening copy.
 *
 * @property rows the number of rows.
 * @property cols the number of columns.
 * @property colPtr column start offsets, length `cols + 1`.
 * @property rowIdx row index of each stored entry, length `values.size`.
 * @property values the stored nonzero values, parallel to [rowIdx].
 */
@Serializable
@SerialName("SparseMatrix")
public class SparseMatrix internal constructor(
    override val rows: Int,
    override val cols: Int,
    public val colPtr: IntArray,
    public val rowIdx: IntArray,
    public val values: DoubleArray,
) : MatrixView {
    init {
        require(rows >= 0 && cols >= 0) { "negative shape: ${rows}x$cols" }
        require(colPtr.size == cols + 1) { "colPtr length ${colPtr.size} != cols+1 ${cols + 1}" }
        require(rowIdx.size == values.size) { "rowIdx/values length mismatch: ${rowIdx.size} vs ${values.size}" }
        require(colPtr[0] == 0) { "colPtr[0] ${colPtr[0]} != 0" }
        require(colPtr[cols] == values.size) { "colPtr[cols] ${colPtr[cols]} != nnz ${values.size}" }
        for (j in 0 until cols) require(colPtr[j] <= colPtr[j + 1]) { "colPtr not monotonic at $j" }
        for (k in rowIdx.indices) require(rowIdx[k] in 0 until rows) { "rowIdx[$k]=${rowIdx[k]} out of [0,$rows)" }
        // Rows must ascend strictly, or the binary search in get reports a stored entry as absent.
        for (j in 0 until cols) {
            for (k in colPtr[j] + 1 until colPtr[j + 1]) {
                require(rowIdx[k - 1] < rowIdx[k]) {
                    "rows must be strictly ascending within a column; column $j has " +
                        "${rowIdx[k - 1]} then ${rowIdx[k]}"
                }
            }
        }
    }

    /** Number of stored nonzeros. */
    public val nnz: Int get() = values.size

    /** Visits the stored entries of column [j] as `(row, value)`, rows ascending. */
    public inline fun forEachInColumn(j: Int, action: (row: Int, value: Double) -> Unit) {
        for (k in colPtr[j] until colPtr[j + 1]) action(rowIdx[k], values[k])
    }

    /**
     * Reads entry (i, j), or `0.0` where nothing is stored. A binary search over the column, so `O(log nnzⱼ)`
     * and fine for a probe; sweep with [forEachInColumn] instead.
     */
    override fun get(i: Int, j: Int): Double {
        require(i in 0 until rows && j in 0 until cols) { "index ($i;$j) outside ${rows}x$cols" }
        var lo = colPtr[j]
        var hi = colPtr[j + 1] - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val row = rowIdx[mid]
            when {
                row < i -> lo = mid + 1
                row > i -> hi = mid - 1
                else -> return values[mid]
            }
        }
        return 0.0
    }

    /** Materialises into a fresh `rows × cols` array of rows; unstored entries stay zero. */
    override fun toArray(): Array<DoubleArray> {
        val out = Array(rows) { DoubleArray(cols) }
        for (j in 0 until cols) forEachInColumn(j) { i, v -> out[i][j] = v }
        return out
    }

    /**
     * Structural equality over the shape and the CSC arrays, so two matrices differing only in which
     * explicit zeros they store are not equal.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SparseMatrix) return false
        return rows == other.rows && cols == other.cols &&
            colPtr.contentEquals(other.colPtr) &&
            rowIdx.contentEquals(other.rowIdx) &&
            values.contentEquals(other.values)
    }

    override fun hashCode(): Int {
        var h = rows * 31 + cols
        h = 31 * h + colPtr.contentHashCode()
        h = 31 * h + rowIdx.contentHashCode()
        h = 31 * h + values.contentHashCode()
        return h
    }

    override fun toString(): String = "SparseMatrix(${rows}x$cols, nnz=$nnz)"

    /** Factories for sparse matrices. */
    public companion object {
        /**
         * Builds a CSC matrix from column-major `(row, value)` entries, where columns(j) lists column j's
         * nonzeros in any order. Entries are sorted by row and duplicate positions are summed.
         */
        public fun ofColumns(rows: Int, cols: Int, columns: List<List<Pair<Int, Double>>>): SparseMatrix {
            require(columns.size == cols) { "expected $cols columns, got ${columns.size}" }
            var nnz = 0
            for (column in columns) nnz += column.size
            // Flattened into triplets, so nothing boxes beyond the pairs the caller already holds.
            val rowIdx = IntArray(nnz)
            val colIdx = IntArray(nnz)
            val values = DoubleArray(nnz)
            var k = 0
            for (j in 0 until cols) {
                for ((i, v) in columns[j]) {
                    rowIdx[k] = i
                    colIdx[k] = j
                    values[k] = v
                    k++
                }
            }
            return ofTriplets(rows, cols, rowIdx, colIdx, values)
        }

        /**
         * Builds a CSC matrix from parallel coordinate (triplet) arrays, where entry k is values(k) at
         * (rowIdx(k), colIdx(k)). Any order, duplicate positions summed, inputs copied. `O(nnz + rows + cols)`.
         */
        public fun ofTriplets(
            rows: Int,
            cols: Int,
            rowIdx: IntArray,
            colIdx: IntArray,
            values: DoubleArray,
        ): SparseMatrix {
            require(rows >= 0 && cols >= 0) { "negative shape: ${rows}x$cols" }
            require(rowIdx.size == colIdx.size && colIdx.size == values.size) {
                "rowIdx/colIdx/values must align: ${rowIdx.size}, ${colIdx.size}, ${values.size}"
            }
            val nnz = values.size
            for (k in 0 until nnz) {
                require(rowIdx[k] in 0 until rows) { "rowIdx[$k]=${rowIdx[k]} out of [0,$rows)" }
                require(colIdx[k] in 0 until cols) { "colIdx[$k]=${colIdx[k]} out of [0,$cols)" }
            }

            // Group by row, so that rowStart(i) is where row i's entries begin once scattered.
            val rowStart = IntArray(rows + 1)
            for (k in 0 until nnz) rowStart[rowIdx[k] + 1]++
            for (i in 0 until rows) rowStart[i + 1] += rowStart[i]
            val byRowCol = IntArray(nnz)
            val byRowVal = DoubleArray(nnz)
            val rowCursor = rowStart.copyOf()
            for (k in 0 until nnz) {
                val p = rowCursor[rowIdx[k]]++
                byRowCol[p] = colIdx[k]
                byRowVal[p] = values[k]
            }

            // Then by column, visiting rows in ascending order, so each column comes out ascending by row.
            val colPtr = IntArray(cols + 1)
            for (k in 0 until nnz) colPtr[byRowCol[k] + 1]++
            for (j in 0 until cols) colPtr[j + 1] += colPtr[j]
            val outRow = IntArray(nnz)
            val outVal = DoubleArray(nnz)
            val colCursor = colPtr.copyOf()
            for (i in 0 until rows) {
                for (k in rowStart[i] until rowStart[i + 1]) {
                    val p = colCursor[byRowCol[k]]++
                    outRow[p] = i
                    outVal[p] = byRowVal[k]
                }
            }

            // Duplicates are now adjacent within a column, so summing them is one forward pass in place.
            val outPtr = IntArray(cols + 1)
            var n = 0
            for (j in 0 until cols) {
                outPtr[j] = n
                var k = colPtr[j]
                while (k < colPtr[j + 1]) {
                    val row = outRow[k]
                    var sum = outVal[k]
                    k++
                    while (k < colPtr[j + 1] && outRow[k] == row) {
                        sum += outVal[k]
                        k++
                    }
                    outRow[n] = row
                    outVal[n] = sum
                    n++
                }
            }
            outPtr[cols] = n
            return SparseMatrix(rows, cols, outPtr, outRow.copyOf(n), outVal.copyOf(n))
        }

        /**
         * Wraps arrays already in CSC form without copying; the caller relinquishes ownership. Validates the
         * invariants rather than repairing them, so use [ofColumns] or [ofTriplets] when they do not hold.
         */
        public fun wrap(rows: Int, cols: Int, colPtr: IntArray, rowIdx: IntArray, values: DoubleArray): SparseMatrix =
            SparseMatrix(rows, cols, colPtr, rowIdx, values)
    }
}
