package com.eignex.koblas

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import kotlin.jvm.JvmStatic

/**
 * Compressed-sparse-column form: column j occupies colPointers(j) until colPointers(j + 1) of
 * [rowIndices] and [values], rows strictly ascending. A stored zero is preserved, and a 64-bit-index
 * host library needs a widening copy.
 *
 * @property rows the number of rows.
 * @property cols the number of columns.
 * Use [copyColumnPointers], [copyRowIndices], or [forEachInColumn] for safe structural access. [colPointers] and
 * [rowIndices] are live zero-copy escape hatches for specialized kernels and require [UnsafeKoblasApi]; mutating
 * them can invalidate the CSC structure. [values] remains live so coefficients can be updated without
 * rebuilding the pattern; do not use the matrix as a hash-map key while mutating it.
 *
 * @property colPointers live column start offsets, length `cols + 1`; do not mutate.
 * @property rowIndices live row index of each stored entry, length `values.size`; do not mutate.
 * @property values the stored values, parallel to the row indices.
 */
@Serializable
@SerialName("SparseMatrix")
public class SparseMatrix internal constructor(
    override val rows: Int,
    override val cols: Int,
    @property:UnsafeKoblasApi public val colPointers: IntArray,
    @property:UnsafeKoblasApi public val rowIndices: IntArray,
    public val values: DoubleArray,
    @Transient private val trustedPattern: Boolean = false,
) : MatrixStorage {
    init {
        requireNonNegativeShape(rows, cols)
        requireShape(colPointers.size.toLong() == cols.toLong() + 1) {
            "colPointers length ${colPointers.size} != cols+1 ${cols.toLong() + 1}"
        }
        requireShape(rowIndices.size == values.size) {
            "rowIndices/values length mismatch: ${rowIndices.size} vs ${values.size}"
        }
        require(colPointers[0] == 0) { "colPointers[0] ${colPointers[0]} != 0" }
        require(colPointers[cols] == values.size) { "colPointers[cols] ${colPointers[cols]} != nnz ${values.size}" }
        for (j in 0 until cols) require(colPointers[j] <= colPointers[j + 1]) { "colPointers not monotonic at $j" }
        // The two passes over rowIndices, which is where an O(nnz) construction spends its checking. A producer
        // deriving this from a matrix that already holds the invariant skips them through [wrapTrusted];
        // everything reaching koblas from outside, a native library above all, still comes through here.
        if (!trustedPattern) {
            for (k in rowIndices.indices) {
                requireIndex(rowIndices[k] in 0 until rows) { "rowIndices[$k]=${rowIndices[k]} out of [0,$rows)" }
            }
            // Rows must ascend strictly, or the binary search in get reports a stored entry as absent.
            for (j in 0 until cols) {
                for (k in colPointers[j] + 1 until colPointers[j + 1]) {
                    require(rowIndices[k - 1] < rowIndices[k]) {
                        "rows must be strictly ascending within a column; column $j has " +
                            "${rowIndices[k - 1]} then ${rowIndices[k]}"
                    }
                }
            }
        }
    }

    /** Number of stored nonzeros. */
    public val nnz: Int get() = values.size

    /** Visits the stored entries of column [j] as `(row, value)`, rows ascending. */
    @kotlin.jvm.JvmSynthetic
    public inline fun forEachInColumn(j: Int, action: (row: Int, value: Double) -> Unit) {
        if (j !in 0 until cols) throw IndexOutOfBoundsException("index $j outside [0,$cols)")
        for (k in colPointers[j] until colPointers[j + 1]) action(rowIndices[k], values[k])
    }

    /**
     * Reads entry (i, j), or `0.0` where nothing is stored. A binary search over the column, so `O(log nnzⱼ)`
     * and fine for a probe; sweep with [forEachInColumn] instead.
     */
    override fun get(i: Int, j: Int): Double {
        requireInBounds(i, j, rows, cols)
        var lo = colPointers[j]
        var hi = colPointers[j + 1] - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val row = rowIndices[mid]
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

    /** A copy of the CSC column start offsets, of length `cols + 1`. */
    public fun copyColumnPointers(): IntArray = colPointers.copyOf()

    /** A copy of the stored row indices, parallel to [values]. */
    public fun copyRowIndices(): IntArray = rowIndices.copyOf()

    /**
     * Structural equality over the shape and the CSC arrays, so two matrices differing only in which
     * explicit zeros they store are not equal.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SparseMatrix) return false
        return rows == other.rows && cols == other.cols &&
            colPointers.contentEquals(other.colPointers) &&
            rowIndices.contentEquals(other.rowIndices) &&
            values.contentEquals(other.values)
    }

    override fun hashCode(): Int {
        var h = rows * 31 + cols
        h = 31 * h + colPointers.contentHashCode()
        h = 31 * h + rowIndices.contentHashCode()
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
        @JvmStatic
        public fun ofColumns(rows: Int, cols: Int, columns: List<List<Pair<Int, Double>>>): SparseMatrix {
            requireNonNegativeShape(rows, cols)
            requireShape(columns.size == cols) { "expected $cols columns, got ${columns.size}" }
            var nnzLong = 0L
            for (column in columns) nnzLong += column.size
            requireShape(nnzLong <= Int.MAX_VALUE) { "stored entry count $nnzLong exceeds Int capacity" }
            val nnz = nnzLong.toInt()
            // Flattened into triplets, so nothing boxes beyond the pairs the caller already holds.
            val rowIndices = IntArray(nnz)
            val colIndices = IntArray(nnz)
            val values = DoubleArray(nnz)
            var k = 0
            for (j in 0 until cols) {
                for ((i, v) in columns[j]) {
                    rowIndices[k] = i
                    colIndices[k] = j
                    values[k] = v
                    k++
                }
            }
            return ofTriplets(rows, cols, rowIndices, colIndices, values)
        }

        /**
         * Builds a CSC matrix from parallel coordinate (triplet) arrays, where entry k is values(k) at
         * (rowIndices(k), colIndices(k)). Any order, duplicate positions summed, inputs copied. `O(nnz + rows + cols)`.
         */
        @JvmStatic
        public fun ofTriplets(
            rows: Int,
            cols: Int,
            rowIndices: IntArray,
            colIndices: IntArray,
            values: DoubleArray,
        ): SparseMatrix {
            requireNonNegativeShape(rows, cols)
            requireShape(rowIndices.size == colIndices.size && colIndices.size == values.size) {
                "rowIndices/colIndices/values must align: ${rowIndices.size}, ${colIndices.size}, ${values.size}"
            }
            val nnz = values.size
            for (k in 0 until nnz) {
                requireIndex(rowIndices[k] in 0 until rows) { "rowIndices[$k]=${rowIndices[k]} out of [0,$rows)" }
                requireIndex(colIndices[k] in 0 until cols) { "colIndices[$k]=${colIndices[k]} out of [0,$cols)" }
            }

            // Group by row, so that rowStart(i) is where row i's entries begin once scattered.
            val rowStart = IntArray(rows + 1)
            for (k in 0 until nnz) rowStart[rowIndices[k] + 1]++
            for (i in 0 until rows) rowStart[i + 1] += rowStart[i]
            val byRowCol = IntArray(nnz)
            val byRowVal = DoubleArray(nnz)
            val rowCursor = rowStart.copyOf()
            for (k in 0 until nnz) {
                val p = rowCursor[rowIndices[k]]++
                byRowCol[p] = colIndices[k]
                byRowVal[p] = values[k]
            }

            // Then by column, visiting rows in ascending order, so each column comes out ascending by row.
            val colPointers = IntArray(cols + 1)
            for (k in 0 until nnz) colPointers[byRowCol[k] + 1]++
            for (j in 0 until cols) colPointers[j + 1] += colPointers[j]
            val outRow = IntArray(nnz)
            val outVal = DoubleArray(nnz)
            val colCursor = colPointers.copyOf()
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
                var k = colPointers[j]
                while (k < colPointers[j + 1]) {
                    val row = outRow[k]
                    var sum = outVal[k]
                    k++
                    while (k < colPointers[j + 1] && outRow[k] == row) {
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
         * invariants rather than repairing them, so use [ofColumns] or [ofTriplets] when they do not hold. The
         * structural arrays cannot be recovered for mutation afterwards.
         */
        @JvmStatic
        public fun wrap(
            rows: Int,
            cols: Int,
            colPointers: IntArray,
            rowIndices: IntArray,
            values: DoubleArray,
        ): SparseMatrix = SparseMatrix(rows, cols, colPointers, rowIndices, values)

        /**
         * The same without the two passes over [rowIndices], for a producer whose output holds the pattern
         * invariant by construction.
         *
         * Only for arrays derived from a matrix that already holds it: a column copied from one, or a
         * pattern this library sorted itself. Arrays reaching koblas from outside go through [wrap], which
         * is what admits an unsorted CSC at all. The shape and pointer checks are cheap and still run.
         */
        internal fun wrapTrusted(
            rows: Int,
            cols: Int,
            colPointers: IntArray,
            rowIndices: IntArray,
            values: DoubleArray,
        ): SparseMatrix = SparseMatrix(rows, cols, colPointers, rowIndices, values, trustedPattern = true)
    }
}
