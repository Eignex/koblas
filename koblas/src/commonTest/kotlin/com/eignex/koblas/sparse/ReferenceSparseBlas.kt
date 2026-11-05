@file:Suppress("VariableNaming", "FunctionParameterNaming", "TooManyFunctions") // math and the sparse surface

package com.eignex.koblas.sparse

import com.eignex.koblas.*

/**
 * Independent sparse oracle using stored coordinates from [SparseMatrix.forEachInColumn].
 * Absent positions form no product; stored zeros participate. Dense expansion would lose that
 * distinction for non-finite values.
 */
internal object ReferenceSparseBlas {

    /** One stored coordinate of a sparse operand. */
    private data class Entry(val row: Int, val col: Int, val value: Double)

    /** Every stored coordinate of [a], in ascending column then row order. */
    private fun entries(a: SparseMatrix): List<Entry> = buildList {
        for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> add(Entry(i, j, v)) }
    }

    /** [a] transposed, which for a coordinate list is a swap of the two indices. */
    fun transpose(a: SparseMatrix): SparseMatrix =
        build(a.cols, a.rows, entries(a).map { Entry(it.col, it.row, it.value) })

    /** The coordinates of `op(A)`, which is where every transpose flag below is resolved. */
    private fun oriented(a: SparseMatrix, transpose: Boolean): List<Entry> =
        if (transpose) entries(a).map { Entry(it.col, it.row, it.value) } else entries(a)

    private fun scaled(beta: Double, previous: Double): Double = if (beta == 0.0) 0.0 else beta * previous

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

    /**
     * Builds sorted CSC arrays independently of [SparseMatrix.ofTriplets], so the oracle does not
     * inherit the production builder's ordering or duplicate handling.
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
