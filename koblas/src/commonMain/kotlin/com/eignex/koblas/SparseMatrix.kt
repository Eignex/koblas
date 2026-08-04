package com.eignex.koblas

import com.eignex.koblas.sparse.SparseLu
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * A sparse matrix in compressed-sparse-column (CSC) form: column `j` occupies
 * `rowIdx[colPtr[j] until colPtr[j + 1]]` with the parallel nonzero values in [values], row indices
 * ascending within a column. `colPtr` has length `cols + 1` with `colPtr[0] == 0` and
 * `colPtr[cols] == values.size`. CSC is the layout sparse solvers and column-oriented sweeps
 * (matrix–vector products, LU factorization) consume directly — and, since [DenseMatrix] became
 * column-major, the axis both storages agree is contiguous.
 *
 * A [MatrixView] like the dense one, so anything written against the view contract accepts either. The
 * two access costs differ though, and the difference is not small: [get] searches a column rather than
 * indexing, and [toArray] densifies. Code that wants the sparsity should reach for [forEachInColumn].
 *
 * Serializes through its CSC arrays rather than a readable 2D form, unlike [DenseMatrix]. The flat dense
 * backing is an implementation detail worth hiding behind a nicer wire shape; CSC is the format, and
 * writing a sparse matrix out densely would cost the `rows × cols` the representation exists to avoid.
 *
 * @property rows the number of rows.
 * @property cols the number of columns.
 * @property colPtr column start offsets, length `cols + 1`.
 * @property rowIdx row index of each stored entry, length `values.size`.
 * @property values the stored nonzero values, parallel to [rowIdx].
 */
@Serializable
@SerialName("SparseMatrix")
class SparseMatrix(
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
