@file:Suppress("VariableNaming", "FunctionParameterNaming", "TooManyFunctions") // math and the sparse surface

package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix

/**
 * A naive oracle for the sparse surface, defined by traversal over stored entries rather than by any kernel.
 *
 * Each operand is a list of stored coordinates pulled through the public [SparseMatrix.forEachInColumn], and
 * nothing here reuses an accumulator, an epoch, a panel or a transpose. Sparse semantics are stated rather
 * than inherited from a dense oracle: a position no operand stores is never visited, so it forms no product
 * and cannot turn an infinity into a NaN, and a stored zero is visited like any other entry.
 */
internal object ReferenceSparseBlas : SparseKernels by BuiltinEngines.scalar.sparseKernels {

    /** One stored coordinate of a sparse operand. */
    data class Entry(val row: Int, val col: Int, val value: Double)

    /** Every stored coordinate of [a], in ascending column then row order. */
    fun entries(a: SparseMatrix): List<Entry> = buildList {
        for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> add(Entry(i, j, v)) }
    }

    /** [a] transposed, which for a coordinate list is a swap of the two indices. */
    fun transpose(a: SparseMatrix): SparseMatrix =
        build(a.cols, a.rows, entries(a).map { Entry(it.col, it.row, it.value) })

    /** The coordinates of `op(A)`, which is where every transpose flag below is resolved. */
    private fun oriented(a: SparseMatrix, transpose: Boolean): List<Entry> =
        if (transpose) entries(a).map { Entry(it.col, it.row, it.value) } else entries(a)

    private fun rowsOf(a: SparseMatrix, transpose: Boolean) = if (transpose) a.cols else a.rows

    private fun colsOf(a: SparseMatrix, transpose: Boolean) = if (transpose) a.rows else a.cols

    private fun scaled(beta: Double, previous: Double): Double = if (beta == 0.0) 0.0 else beta * previous

    /* A discovered position as one map key, so the oracle accumulates coordinates without nesting maps. */
    private fun coordinate(row: Int, col: Int): Long = (row.toLong() shl Int.SIZE_BITS) or col.toLong()

    private fun rowOf(key: Long): Int = (key ushr Int.SIZE_BITS).toInt()

    private fun colOf(key: Long): Int = key.toInt()

    fun gemv(alpha: Double, a: SparseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false) {
        for (i in y.indices) y[i] = scaled(beta, y[i])
        if (alpha == 0.0) return
        for (e in oriented(a, transpose)) y[e.row] += alpha * (e.value * x[e.col])
    }

    /**
     * The symmetric product, defined by mirroring the selected triangle's stored entries. Exactly the
     * coordinates of that triangle are visited, and a stored off-diagonal entry contributes twice.
     */
    fun symv(alpha: Double, a: SparseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean = true) {
        for (i in y.indices) y[i] = scaled(beta, y[i])
        if (alpha == 0.0) return
        for (e in selectedTriangle(a, lower)) {
            y[e.row] += alpha * (e.value * x[e.col])
            if (e.row != e.col) y[e.col] += alpha * (e.value * x[e.row])
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymm signature plus the side
    fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
    ) {
        for (i in c.values.indices) c.values[i] = scaled(beta, c.values[i])
        if (alpha == 0.0) return
        for (e in selectedTriangle(a, lower)) {
            mirrorInto(alpha, e.row, e.col, e.value, b, c, right)
            if (e.row != e.col) mirrorInto(alpha, e.col, e.row, e.value, b, c, right)
        }
    }

    private fun mirrorInto(
        alpha: Double,
        i: Int,
        j: Int,
        value: Double,
        b: DenseMatrix,
        c: DenseMatrix,
        right: Boolean,
    ) {
        if (right) {
            for (row in 0 until c.rows) c.values[row + j * c.rows] += alpha * (b.values[row + i * b.rows] * value)
        } else {
            for (col in 0 until c.cols) c.values[i + col * c.rows] += alpha * (value * b.values[j + col * b.rows])
        }
    }

    private fun selectedTriangle(a: SparseMatrix, lower: Boolean): List<Entry> =
        entries(a).filter { if (lower) it.row >= it.col else it.row <= it.col }

    /** `C = alpha·op(A)·op(B) + beta·C` for a sparse `A` and dense `B`, or the mirrored product when [right]. */
    @Suppress("LongParameterList") // the BLAS dgemm signature plus the side
    fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        right: Boolean = false,
    ) {
        for (i in c.values.indices) c.values[i] = scaled(beta, c.values[i])
        if (alpha == 0.0) return
        val dense = { i: Int, j: Int -> if (transposeB) b.values[j + i * b.rows] else b.values[i + j * b.rows] }
        for (e in oriented(a, transposeA)) {
            if (right) {
                for (row in 0 until c.rows) c.values[row + e.col * c.rows] += alpha * (dense(row, e.row) * e.value)
            } else {
                for (col in 0 until c.cols) c.values[e.row + col * c.rows] += alpha * (e.value * dense(e.col, col))
            }
        }
    }

    /** `alpha·op(A)·op(B)` for two sparse operands, into a fresh CSC result. */
    fun gemm(alpha: Double, a: SparseMatrix, transposeA: Boolean, b: SparseMatrix, transposeB: Boolean): SparseMatrix {
        val left = oriented(a, transposeA)
        val right = oriented(b, transposeB)
        val discovered = LinkedHashMap<Long, Double>()
        for (l in left) {
            for (r in right) {
                if (r.row != l.col) continue
                val key = coordinate(l.row, r.col)
                discovered[key] = (discovered[key] ?: 0.0) + l.value * r.value
            }
        }
        val entries = discovered.map { (key, sum) ->
            Entry(rowOf(key), colOf(key), if (alpha == 0.0) alpha else alpha * sum)
        }
        return build(rowsOf(a, transposeA), colsOf(b, transposeB), entries)
    }

    /** The same product accumulated into a dense destination. */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ) {
        for (i in c.values.indices) c.values[i] = scaled(beta, c.values[i])
        if (alpha == 0.0) return
        for (l in oriented(a, transposeA)) {
            for (r in oriented(b, transposeB)) {
                if (r.row != l.col) continue
                c.values[l.row + r.col * c.rows] += alpha * (l.value * r.value)
            }
        }
    }

    /** `C = alpha·op(A)·op(A)ᵀ + beta·C` over the selected triangle of a dense destination. */
    @Suppress("LongParameterList") // the BLAS dsyrk signature
    fun syrk(alpha: Double, a: SparseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, lower: Boolean = true) {
        val n = c.rows
        for (j in 0 until n) {
            for (i in 0 until n) {
                if (if (lower) i >= j else i <= j) c.values[i + j * n] = scaled(beta, c.values[i + j * n])
            }
        }
        if (alpha == 0.0) return
        val op = oriented(a, transpose)
        for (l in op) {
            for (r in op) {
                if (l.col != r.col) continue
                if (if (lower) l.row < r.row else l.row > r.row) continue
                c.values[l.row + r.row * n] += alpha * (l.value * r.value)
            }
        }
    }

    /** The selected triangle of `op(A)·op(A)ᵀ` as a fresh CSC result. */
    fun syrk(a: SparseMatrix, transpose: Boolean = false, lower: Boolean = true): SparseMatrix {
        val op = oriented(a, transpose)
        val order = rowsOf(a, transpose)
        val discovered = LinkedHashMap<Long, Double>()
        for (l in op) {
            for (r in op) {
                if (l.col != r.col) continue
                if (if (lower) l.row < r.row else l.row > r.row) continue
                val key = coordinate(l.row, r.row)
                discovered[key] = (discovered[key] ?: 0.0) + l.value * r.value
            }
        }
        val entries = discovered.map { (key, sum) ->
            Entry(rowOf(key), colOf(key), sum)
        }
        return build(order, order, entries)
    }

    /** `alpha·op(A) + B` as a fresh CSC result retaining the structural union. */
    fun addScaled(alpha: Double, a: SparseMatrix, transposeA: Boolean, b: SparseMatrix): SparseMatrix {
        val union = LinkedHashMap<Long, Double>()
        for (e in oriented(a, transposeA)) {
            val key = coordinate(e.row, e.col)
            union[key] = (union[key] ?: 0.0) + if (alpha == 0.0) alpha else alpha * e.value
        }
        for (e in entries(b)) {
            val key = coordinate(e.row, e.col)
            union[key] = (union[key] ?: 0.0) + e.value
        }
        val entries = union.map { (key, sum) ->
            Entry(rowOf(key), colOf(key), sum)
        }
        return build(rowsOf(a, transposeA), colsOf(a, transposeA), entries)
    }

    /**
     * A CSC matrix over discovered coordinates, sorted rather than accumulated. [SparseMatrix.ofTriplets]
     * would sum duplicates, which the oracle has already done, so this is a second implementation of the
     * ordering the library's builders also have to get right.
     */
    private fun build(rows: Int, cols: Int, entries: List<Entry>): SparseMatrix {
        val sorted = entries.sortedWith(compareBy({ it.col }, { it.row }))
        val pointers = IntArray(cols + 1)
        for (e in sorted) pointers[e.col + 1]++
        for (j in 0 until cols) pointers[j + 1] += pointers[j]
        return SparseMatrix.wrap(
            rows,
            cols,
            pointers,
            IntArray(sorted.size) { sorted[it].row },
            DoubleArray(sorted.size) { sorted[it].value },
        )
    }
}
