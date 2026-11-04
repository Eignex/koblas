package com.eignex.koblas.bench

import com.eignex.koblas.*

/**
 * What the sparse cases add to [DenseReference] when a case is checked before it is timed.
 *
 * Nothing here shares code with the library's sparse scheduling. Each operand is materialised once through
 * the public column walk and handed to the dense textbook definitions, so the result is the dense sum over the
 * whole shape. That is valid because these fixtures are finite: densifying would get an implicit zero's
 * exceptional behaviour wrong, and that is what the library's own conformance tests cover with hand-computed
 * values instead.
 *
 * The point of checking here is that a benchmark publishing a number has to have computed the right thing.
 * A case whose result disagrees is a failure, not a fast row.
 */
internal object SparseReference {
    /** [a] as dense column-major storage, every unstored position zero. */
    fun dense(a: SparseMatrix): DenseMatrix {
        val result = DenseMatrix.zero(a.rows, a.cols)
        for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> result.values[i + j * a.rows] = v }
        return result
    }

    /** `alpha·A + B` over the whole shape, column-major. */
    fun addScaled(alpha: Double, a: SparseMatrix, b: SparseMatrix): DoubleArray {
        val left = dense(a).values
        val right = dense(b).values
        return DoubleArray(left.size) { alpha * left[it] + right[it] }
    }

    /**
     * The positions a product of two patterns reaches, derived from the patterns alone.
     *
     * Values never enter it: a stored zero selects and scatters like any other entry, and an entry the
     * arithmetic cancels is kept. That is exactly what a numeric comparison cannot see, which is why the
     * support a sparse result claims is checked against this rather than against the values it holds.
     */
    fun productSupport(
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
    ): Array<BooleanArray> {
        val left = pattern(a, transposeA)
        val right = pattern(b, transposeB)
        val depth = if (left.isEmpty()) 0 else left[0].size
        val columns = if (right.isEmpty()) 0 else right[0].size
        return Array(left.size) { i ->
            BooleanArray(columns) { j ->
                (0 until depth).any { p -> left[i][p] && right[p][j] }
            }
        }
    }

    /** The selected triangle of the positions `op(A)·op(A)ᵀ` reaches. */
    fun rankSupport(a: SparseMatrix, transpose: Boolean, lower: Boolean): Array<BooleanArray> {
        val op = pattern(a, transpose)
        val depth = if (op.isEmpty()) 0 else op[0].size
        return Array(op.size) { i ->
            BooleanArray(op.size) { j ->
                selected(i, j, lower) && (0 until depth).any { p -> op[i][p] && op[j][p] }
            }
        }
    }

    /** Whether position `(i, j)` lies in the selected triangle, diagonal included. */
    private fun selected(i: Int, j: Int, lower: Boolean): Boolean = if (lower) i >= j else i <= j

    /** The union of two patterns, which is what a scaled structural addition retains. */
    fun unionSupport(a: SparseMatrix, transposeA: Boolean, b: SparseMatrix): Array<BooleanArray> {
        val left = pattern(a, transposeA)
        val right = pattern(b, false)
        val columns = if (left.isEmpty()) 0 else left[0].size
        return Array(left.size) { i -> BooleanArray(columns) { j -> left[i][j] || right[i][j] } }
    }

    /** The stored positions of [a] in the requested orientation, read through the public column walk. */
    private fun pattern(a: SparseMatrix, transpose: Boolean): Array<BooleanArray> {
        val rows = if (transpose) a.cols else a.rows
        val columns = if (transpose) a.rows else a.cols
        val stored = Array(rows) { BooleanArray(columns) }
        for (j in 0 until a.cols) {
            a.forEachInColumn(j) { i, _ -> if (transpose) stored[j][i] = true else stored[i][j] = true }
        }
        return stored
    }

    /**
     * Fails when a fresh CSC result is not the column-major matrix [expected] describes, or is not valid CSC.
     *
     * Two checks, because a sparse result has two halves. Materialising it catches a wrong or missing value
     * anywhere in the shape; walking its pointers catches a support that is out of order or inconsistent with
     * what it claims to hold, which materialising would hide.
     */
    fun checkSparse(
        expected: DoubleArray,
        support: Array<BooleanArray>,
        actual: SparseMatrix,
        what: String,
    ) {
        val pointers = actual.copyColumnPointers()
        val indices = actual.copyRowIndices()
        check(pointers.size == actual.cols + 1) { "$what: ${pointers.size} pointers for ${actual.cols} columns" }
        check(pointers[actual.cols] == actual.nnz) { "$what: pointers end at ${pointers[actual.cols]}, nnz ${actual.nnz}" }
        for (j in 0 until actual.cols) {
            check(pointers[j] <= pointers[j + 1]) { "$what: column $j pointers descend" }
            for (k in pointers[j] + 1 until pointers[j + 1]) {
                check(indices[k - 1] < indices[k]) { "$what: column $j rows do not ascend at $k" }
            }
        }
        DenseReference.check(expected, dense(actual).values, what)
        checkSupport(support, pointers, indices, actual, what)
    }

    /**
     * Fails when the positions a result stores are not the ones its operands' patterns reach.
     *
     * Values cannot answer this. A position whose arithmetic cancels to zero must be stored and a position
     * neither operand reaches must not be, and both look like a zero from the outside.
     */
    private fun checkSupport(
        support: Array<BooleanArray>,
        pointers: IntArray,
        indices: IntArray,
        actual: SparseMatrix,
        what: String,
    ) {
        check(support.size == actual.rows) { "$what: support has ${support.size} rows, result has ${actual.rows}" }
        val stored = Array(actual.rows) { BooleanArray(actual.cols) }
        for (j in 0 until actual.cols) {
            for (k in pointers[j] until pointers[j + 1]) stored[indices[k]][j] = true
        }
        for (i in 0 until actual.rows) {
            for (j in 0 until actual.cols) {
                check(support[i][j] == stored[i][j]) {
                    val expectation = if (support[i][j]) "stored" else "absent"
                    "$what: position ($i, $j) should be $expectation"
                }
            }
        }
    }
}
