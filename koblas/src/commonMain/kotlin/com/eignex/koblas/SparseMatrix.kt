package com.eignex.koblas

/**
 * A sparse matrix in compressed-sparse-column (CSC) form: column `j` occupies
 * `rowIdx[colPtr[j] until colPtr[j + 1]]` with the parallel nonzero values in [values], row indices
 * ascending within a column. `colPtr` has length `cols + 1` with `colPtr[0] == 0` and
 * `colPtr[cols] == values.size`. CSC is the layout sparse solvers and column-oriented sweeps
 * (matrix–vector products, LU factorization) consume directly.
 *
 * @property rows the number of rows.
 * @property cols the number of columns.
 * @property colPtr column start offsets, length `cols + 1`.
 * @property rowIdx row index of each stored entry, length `values.size`.
 * @property values the stored nonzero values, parallel to [rowIdx].
 */
class SparseMatrix(val rows: Int, val cols: Int, val colPtr: IntArray, val rowIdx: IntArray, val values: DoubleArray) {
    init {
        require(rows >= 0 && cols >= 0) { "negative shape: ${rows}x$cols" }
        require(colPtr.size == cols + 1) { "colPtr length ${colPtr.size} != cols+1 ${cols + 1}" }
        require(rowIdx.size == values.size) { "rowIdx/values length mismatch: ${rowIdx.size} vs ${values.size}" }
        require(colPtr[0] == 0) { "colPtr[0] ${colPtr[0]} != 0" }
        require(colPtr[cols] == values.size) { "colPtr[cols] ${colPtr[cols]} != nnz ${values.size}" }
        for (j in 0 until cols) require(colPtr[j] <= colPtr[j + 1]) { "colPtr not monotonic at $j" }
        for (k in rowIdx.indices) require(rowIdx[k] in 0 until rows) { "rowIdx[$k]=${rowIdx[k]} out of [0,$rows)" }
    }

    /** Number of stored nonzeros. */
    val nnz: Int get() = values.size

    /** Visit the nonzero entries of column [j] as `(row, value)`, rows ascending. */
    inline fun forEachInColumn(j: Int, action: (row: Int, value: Double) -> Unit) {
        for (k in colPtr[j] until colPtr[j + 1]) action(rowIdx[k], values[k])
    }

    /** Matrix-vector product `A · x` (length [rows]), or `Aᵀ · x` (length [cols]) when [transpose]. */
    fun gemv(x: DoubleArray, transpose: Boolean = false): DoubleArray {
        if (transpose) {
            require(x.size == rows) { "gemvᵀ: x length ${x.size} != rows $rows" }
            val out = DoubleArray(cols)
            for (j in 0 until cols) {
                var s = 0.0
                for (k in colPtr[j] until colPtr[j + 1]) s += values[k] * x[rowIdx[k]]
                out[j] = s
            }
            return out
        }
        require(x.size == cols) { "gemv: x length ${x.size} != cols $cols" }
        val out = DoubleArray(rows)
        for (j in 0 until cols) {
            val xj = x[j]
            if (xj != 0.0) for (k in colPtr[j] until colPtr[j + 1]) out[rowIdx[k]] += values[k] * xj
        }
        return out
    }

    /** Build the dense per-row maps [SparseLu.factorize] consumes (row → column → value). */
    fun toRowMaps(): Array<HashMap<Int, Double>> {
        val out = Array(rows) { HashMap<Int, Double>() }
        for (j in 0 until cols) {
            for (k in colPtr[j] until colPtr[j + 1]) out[rowIdx[k]][j] = values[k]
        }
        return out
    }

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
    }
}
