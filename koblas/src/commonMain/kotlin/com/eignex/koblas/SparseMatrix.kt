package com.eignex.koblas

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * A sparse matrix in compressed-sparse-column (CSC) form: column `j` occupies
 * `rowIdx[colPtr[j] until colPtr[j + 1]]` with the parallel values in [values], row indices strictly
 * ascending within a column and validated by the constructor, since [get] binary-searches them.
 *
 * The invariants are UMFPACK's preconditions verbatim, checked against the headers, so this backing crosses
 * to `umfpack_di_*` with no repacking; KLU and CHOLMOD accept the same arrays. A 64-bit-index library would
 * need the `dl` family and a widening copy, and an explicitly stored zero is part of the value here where a
 * host may drop it, so a round trip through one can lose pattern.
 *
 * A [MatrixView] like the dense one, but the access costs differ sharply: [get] searches a column and
 * [toArray] densifies, so code that wants the sparsity should use [forEachInColumn]. Serializes as its CSC
 * arrays, since writing it densely would cost the `rows × cols` the representation exists to avoid.
 *
 * Construction goes through the [Companion] factories, as it does for every koblas container.
 *
 * @property rows the number of rows.
 * @property cols the number of columns.
 * @property colPtr column start offsets, length `cols + 1`.
 * @property rowIdx row index of each stored entry, length `values.size`.
 * @property values the stored nonzero values, parallel to [rowIdx].
 */
@Serializable
@SerialName("SparseMatrix")
class SparseMatrix internal constructor(
    override val rows: Int,
    override val cols: Int,
    val colPtr: IntArray,
    val rowIdx: IntArray,
    val values: DoubleArray,
) : MatrixView {
    init {
        require(rows >= 0 && cols >= 0) { "negative shape: ${rows}x$cols" }
        require(colPtr.size == cols + 1) { "colPtr length ${colPtr.size} != cols+1 ${cols + 1}" }
        require(rowIdx.size == values.size) { "rowIdx/values length mismatch: ${rowIdx.size} vs ${values.size}" }
        require(colPtr[0] == 0) { "colPtr[0] ${colPtr[0]} != 0" }
        require(colPtr[cols] == values.size) { "colPtr[cols] ${colPtr[cols]} != nnz ${values.size}" }
        for (j in 0 until cols) require(colPtr[j] <= colPtr[j + 1]) { "colPtr not monotonic at $j" }
        for (k in rowIdx.indices) require(rowIdx[k] in 0 until rows) { "rowIdx[$k]=${rowIdx[k]} out of [0,$rows)" }
        // Rows strictly ascending within each column. The format has always documented this and the
        // factories have always produced it, but it went unchecked — and [get] binary-searches the column,
        // so an unsorted one would not throw, it would quietly report a stored entry as absent. Strict
        // ascent also rules out duplicate rows, which would otherwise make [get] and [forEachInColumn]
        // disagree about the value at a position.
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
    val nnz: Int get() = values.size

    /** Visit the nonzero entries of column [j] as `(row, value)`, rows ascending. */
    inline fun forEachInColumn(j: Int, action: (row: Int, value: Double) -> Unit) {
        for (k in colPtr[j] until colPtr[j + 1]) action(rowIdx[k], values[k])
    }

    /**
     * Read entry `(i, j)`, or `0.0` where nothing is stored.
     *
     * A binary search over column `j`'s row indices, which are ascending — so this is `O(log nnzⱼ)`, not
     * the `O(1)` the dense storage gives. Fine for a probe; wrong for a sweep, where [forEachInColumn]
     * visits the same information in `O(nnzⱼ)` total.
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

    /** Materialise into a fresh `rows × cols` array of rows. Densifies, so the result is as large as the
     *  representation was avoiding; only the stored entries are written, the rest stay zero. */
    override fun toArray(): Array<DoubleArray> {
        val out = Array(rows) { DoubleArray(cols) }
        for (j in 0 until cols) forEachInColumn(j) { i, v -> out[i][j] = v }
        return out
    }

    /**
     * Structural equality over the shape and the stored entries.
     *
     * Compares the CSC arrays, which means two matrices that agree everywhere but store a different set
     * of explicit zeros are *not* equal. That mirrors [SparseVector], where a stored zero is also part of
     * the value: the pattern is information, and a factorization that reserved a slot for fill differs
     * from one that never had it.
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

    /** Factories for CSC matrices. */
    companion object {
        /** Build a CSC matrix from column-major `(row, value)` entries: `columns[j]` lists column `j`'s
         *  nonzeros. Entries within a column are sorted by row; duplicates are summed. */
        fun ofColumns(rows: Int, cols: Int, columns: List<List<Pair<Int, Double>>>): SparseMatrix {
            require(columns.size == cols) { "expected $cols columns, got ${columns.size}" }
            val colPtr = IntArray(cols + 1)
            val rowIdxList = ArrayList<Int>()
            val valueList = ArrayList<Double>()
            for (j in 0 until cols) {
                colPtr[j] = rowIdxList.size
                val merged = HashMap<Int, Double>()
                for ((i, v) in columns[j]) merged[i] = (merged[i] ?: 0.0) + v
                for (i in merged.keys.sorted()) {
                    rowIdxList.add(i)
                    valueList.add(merged.getValue(i))
                }
            }
            colPtr[cols] = rowIdxList.size
            return SparseMatrix(rows, cols, colPtr, rowIdxList.toIntArray(), valueList.toDoubleArray())
        }

        /**
         * Build a CSC matrix from parallel coordinate (triplet) arrays: entry `k` is `values[k]` at
         * `(rowIdx[k], colIdx[k])`. Entries may arrive in any order and a repeated position contributes
         * the sum of its values, matching [ofColumns] and [SparseVector.of]. Copies its inputs, so the
         * caller can reuse the arrays.
         *
         * The interchange format — Matrix Market, `scipy.sparse.coo_matrix`, Eigen's triplet list — and
         * primitive throughout, where [ofColumns] boxes an `Int`, a `Double`, a `Pair` and a list per
         * entry.
         *
         * Two counting passes, no comparison sort: `O(nnz + rows + cols)`.
         */
        fun ofTriplets(rows: Int, cols: Int, rowIdx: IntArray, colIdx: IntArray, values: DoubleArray): SparseMatrix {
            require(rows >= 0 && cols >= 0) { "negative shape: ${rows}x$cols" }
            require(rowIdx.size == colIdx.size && colIdx.size == values.size) {
                "rowIdx/colIdx/values must align: ${rowIdx.size}, ${colIdx.size}, ${values.size}"
            }
            val nnz = values.size
            for (k in 0 until nnz) {
                require(rowIdx[k] in 0 until rows) { "rowIdx[$k]=${rowIdx[k]} out of [0,$rows)" }
                require(colIdx[k] in 0 until cols) { "colIdx[$k]=${colIdx[k]} out of [0,$cols)" }
            }

            // Group by row: rowStart[i] is where row i's entries begin once scattered.
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

            // Then by column, visiting rows in ascending order, so each column comes out ascending by
            // row without ever comparing two entries.
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

            // Duplicates are now adjacent within a column, so summing them is one forward pass, in
            // place: the write position trails the read position by the number already merged.
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
         * Wrap arrays that are already in CSC form without copying. Caller relinquishes ownership.
         *
         * Validates the invariants rather than repairing them, so [rowIdx] must already be strictly
         * ascending within each column. Use [ofColumns] or [ofTriplets] when it is not.
         */
        fun wrap(rows: Int, cols: Int, colPtr: IntArray, rowIdx: IntArray, values: DoubleArray): SparseMatrix =
            SparseMatrix(rows, cols, colPtr, rowIdx, values)
    }
}
