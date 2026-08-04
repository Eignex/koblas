package com.eignex.koblas

import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.sparseKoblas
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * A sparse matrix in compressed-sparse-column (CSC) form: column `j` occupies
 * `rowIdx[colPtr[j] until colPtr[j + 1]]` with the parallel nonzero values in [values], row indices
 * strictly ascending within a column — validated by the constructor, since [get] relies on the
 * ordering. `colPtr` has length `cols + 1` with `colPtr[0] == 0` and
 * `colPtr[cols] == values.size`. CSC is the layout sparse solvers and column-oriented sweeps
 * (matrix–vector products, LU factorization) consume directly — and, since [DenseMatrix] became
 * column-major, the axis both storages agree is contiguous.
 *
 * The invariants are not arbitrary: they are what a host sparse solver requires. UMFPACK states them
 * verbatim — `Ap[0]` zero, `Ap[j] <= Ap[j+1]`, row indices ascending within a column with no duplicates,
 * 0-based and in range — so this backing passes to `umfpack_di_*` with no repacking. KLU takes the same
 * `(n, Ap, Ai)` with `int32_t` arrays. CHOLMOD wraps it as a `cholmod_sparse` with `packed = 1`,
 * `sorted = 1`, `stype = 0` (both triangles stored, which is what koblas does) and `itype = CHOLMOD_INT`.
 * Checked against the 7.x headers rather than remembered.
 *
 * Two mismatches remain for a future binding, neither structural. A library built for 64-bit indices
 * wants the `umfpack_dl_*` family and a widening copy, the sparse counterpart of the LP64/ILP64 split the
 * dense backend already documents. And an explicitly stored zero is part of the value here — equality
 * distinguishes it — where a host library may drop it, so a round-trip through one can lose pattern.
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

    /**
     * Matrix-vector product `A · x` (length [rows]), or `Aᵀ · x` (length [cols]) when [transpose].
     *
     * The member spelling of [SparseBlas.gemv], forwarding to the active backend — the same arrangement
     * the dense side uses, where `trsv` and friends exist as both interface members and free functions so
     * a call site can read whichever way suits it. The arithmetic lives on the seam, not here.
     */
    fun gemv(x: DoubleArray, transpose: Boolean = false): DoubleArray = sparseKoblas.gemv(this, x, transpose)

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
