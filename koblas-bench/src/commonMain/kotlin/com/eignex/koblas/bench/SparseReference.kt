@file:Suppress("VariableNaming", "FunctionParameterNaming", "TooManyFunctions") // math and the sparse surface

package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import kotlin.math.abs

/**
 * An explicit scalar reference for the sparse cases, used to check a case before it is timed.
 *
 * Nothing here shares code with the library's sparse scheduling. Each operand is materialised through the
 * public [SparseMatrix.toArray], and the result is the textbook dense sum over the whole shape. That is valid
 * because these fixtures are finite: densifying would get an implicit zero's exceptional behaviour wrong, and
 * that is what the library's own conformance tests cover with hand-computed values instead.
 *
 * The point of checking here is that a benchmark publishing a number has to have computed the right thing.
 * A case whose result disagrees is a failure, not a fast row.
 */
internal object SparseReference {
    /** Relative tolerance for a dense sum against a CSC traversal of the same terms in a different order. */
    private const val TOLERANCE = 1e-9

    fun at(a: Array<DoubleArray>, i: Int, j: Int, transpose: Boolean): Double =
        if (transpose) a[j][i] else a[i][j]

    fun rows(a: Array<DoubleArray>, transpose: Boolean): Int = if (transpose) columnsOf(a) else a.size

    fun columns(a: Array<DoubleArray>, transpose: Boolean): Int = if (transpose) a.size else columnsOf(a)

    private fun columnsOf(a: Array<DoubleArray>): Int = if (a.isEmpty()) 0 else a[0].size

    fun dense(m: DenseMatrix): Array<DoubleArray> = Array(m.rows) { i -> DoubleArray(m.cols) { j -> m[i, j] } }

    /** `y = alpha·op(A)·x + beta·y`, over the whole shape. */
    fun gemv(
        alpha: Double,
        a: Array<DoubleArray>,
        transpose: Boolean,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
    ): DoubleArray {
        val m = rows(a, transpose)
        val k = columns(a, transpose)
        return DoubleArray(m) { i ->
            var sum = 0.0
            for (p in 0 until k) sum += at(a, i, p, transpose) * x[p]
            alpha * sum + if (beta == 0.0) 0.0 else beta * y[i]
        }
    }

    /** `C = alpha·op(A)·op(B) + beta·C`, over the whole shape, with the operands already in dense form. */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    fun gemm(
        alpha: Double,
        a: Array<DoubleArray>,
        transposeA: Boolean,
        b: Array<DoubleArray>,
        transposeB: Boolean,
        beta: Double,
        c: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val m = rows(a, transposeA)
        val depth = columns(a, transposeA)
        val n = columns(b, transposeB)
        return Array(m) { i ->
            DoubleArray(n) { j ->
                var sum = 0.0
                for (p in 0 until depth) sum += at(a, i, p, transposeA) * at(b, p, j, transposeB)
                alpha * sum + if (beta == 0.0) 0.0 else beta * c[i][j]
            }
        }
    }

    /** The full symmetric matrix the selected triangle of [a] stands for. */
    fun mirrored(a: Array<DoubleArray>, lower: Boolean): Array<DoubleArray> {
        val n = a.size
        return Array(n) { i ->
            DoubleArray(n) { j ->
                if (lower == (i >= j)) a[i][j] else a[j][i]
            }
        }
    }

    /** The selected triangle of [a], with the rest zero and an implicit unit diagonal supplied when asked. */
    fun triangle(a: Array<DoubleArray>, lower: Boolean, unitDiag: Boolean): Array<DoubleArray> {
        val n = a.size
        return Array(n) { i ->
            DoubleArray(n) { j ->
                when {
                    i == j && unitDiag -> 1.0
                    lower && i >= j -> a[i][j]
                    !lower && i <= j -> a[i][j]
                    else -> 0.0
                }
            }
        }
    }

    /** `x = op(T)·x` over an explicit triangle. */
    fun trmv(t: Array<DoubleArray>, transpose: Boolean, x: DoubleArray): DoubleArray {
        val n = x.size
        return DoubleArray(n) { i ->
            var sum = 0.0
            for (j in 0 until n) sum += at(t, i, j, transpose) * x[j]
            sum
        }
    }

    /** `op(T)·x = b` by substitution over an explicit triangle. */
    fun trsv(t: Array<DoubleArray>, lower: Boolean, transpose: Boolean, b: DoubleArray): DoubleArray {
        val n = b.size
        val x = b.copyOf()
        val forward = lower != transpose
        val order = if (forward) 0 until n else n - 1 downTo 0
        for (i in order) {
            var sum = x[i]
            val inner = if (forward) 0 until i else i + 1 until n
            for (j in inner) sum -= at(t, i, j, transpose) * x[j]
            x[i] = sum / at(t, i, i, transpose)
        }
        return x
    }

    /** `B = alpha·op(T)·B`, or `B = alpha·B·op(T)` when [right], over an explicit triangle. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    fun trmm(
        t: Array<DoubleArray>,
        transpose: Boolean,
        right: Boolean,
        alpha: Double,
        b: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val scaled = Array(b.size) { i -> DoubleArray(b[i].size) { j -> alpha * b[i][j] } }
        if (scaled.isEmpty()) return scaled
        return if (right) {
            transposed(trmmColumns(t, !transpose, transposed(scaled)))
        } else {
            trmmColumns(t, transpose, scaled)
        }
    }

    /** `B = alpha·op(T)⁻¹·B`, or the right-hand form, over an explicit triangle. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    fun trsm(
        t: Array<DoubleArray>,
        lower: Boolean,
        transpose: Boolean,
        right: Boolean,
        alpha: Double,
        b: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val scaled = Array(b.size) { i -> DoubleArray(b[i].size) { j -> alpha * b[i][j] } }
        if (scaled.isEmpty()) return scaled
        return if (right) {
            transposed(trsmColumns(t, lower, !transpose, transposed(scaled)))
        } else {
            trsmColumns(t, lower, transpose, scaled)
        }
    }

    private fun trmmColumns(t: Array<DoubleArray>, transpose: Boolean, b: Array<DoubleArray>): Array<DoubleArray> {
        val columns = if (b.isEmpty()) 0 else b[0].size
        val out = Array(b.size) { DoubleArray(columns) }
        for (j in 0 until columns) {
            val column = trmv(t, transpose, DoubleArray(b.size) { i -> b[i][j] })
            for (i in b.indices) out[i][j] = column[i]
        }
        return out
    }

    private fun trsmColumns(
        t: Array<DoubleArray>,
        lower: Boolean,
        transpose: Boolean,
        b: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val columns = if (b.isEmpty()) 0 else b[0].size
        val out = Array(b.size) { DoubleArray(columns) }
        for (j in 0 until columns) {
            val column = trsv(t, lower, transpose, DoubleArray(b.size) { i -> b[i][j] })
            for (i in b.indices) out[i][j] = column[i]
        }
        return out
    }

    private fun transposed(a: Array<DoubleArray>): Array<DoubleArray> {
        val columns = if (a.isEmpty()) 0 else a[0].size
        return Array(columns) { j -> DoubleArray(a.size) { i -> a[i][j] } }
    }

    /** The selected triangle of `alpha·op(A)·op(A)ᵀ + beta·C`, with the other triangle left as it was. */
    @Suppress("LongParameterList") // the BLAS dsyrk signature
    fun syrk(
        alpha: Double,
        a: Array<DoubleArray>,
        transpose: Boolean,
        beta: Double,
        c: Array<DoubleArray>,
        lower: Boolean,
    ): Array<DoubleArray> {
        val n = rows(a, transpose)
        val depth = columns(a, transpose)
        return Array(n) { i ->
            DoubleArray(n) { j ->
                // The diagonal belongs to whichever triangle was selected, so it is written either way.
                if (!selected(i, j, lower)) {
                    c[i][j]
                } else {
                    var sum = 0.0
                    for (p in 0 until depth) sum += at(a, i, p, transpose) * at(a, j, p, transpose)
                    alpha * sum + if (beta == 0.0) 0.0 else beta * c[i][j]
                }
            }
        }
    }

    /** Whether position `(i, j)` lies in the selected triangle, diagonal included. */
    fun selected(i: Int, j: Int, lower: Boolean): Boolean = if (lower) i >= j else i <= j

    /** `alpha·op(A) + B` over the whole shape. */
    fun addScaled(
        alpha: Double,
        a: Array<DoubleArray>,
        transposeA: Boolean,
        b: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val m = rows(a, transposeA)
        val n = columns(a, transposeA)
        return Array(m) { i -> DoubleArray(n) { j -> alpha * at(a, i, j, transposeA) + b[i][j] } }
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

    /** Fails when [actual] differs from [expected] anywhere, naming the case and the position. */
    fun check(expected: Array<DoubleArray>, actual: Array<DoubleArray>, what: String) {
        check(expected.size == actual.size) { "$what: ${actual.size} rows, expected ${expected.size}" }
        for (i in expected.indices) check(expected[i], actual[i], "$what row $i")
    }

    /** Fails when [actual] differs from [expected] anywhere in one vector. */
    fun check(expected: DoubleArray, actual: DoubleArray, what: String) {
        check(expected.size == actual.size) { "$what: ${actual.size} entries, expected ${expected.size}" }
        for (i in expected.indices) {
            val bound = TOLERANCE * maxOf(1.0, abs(expected[i]))
            check(abs(expected[i] - actual[i]) <= bound) {
                "$what at $i: ${actual[i]}, expected ${expected[i]}"
            }
        }
    }

    /**
     * Fails when a fresh CSC result is not the matrix [expected] describes, or is not valid CSC.
     *
     * Two checks, because a sparse result has two halves. Materialising it catches a wrong or missing value
     * anywhere in the shape; walking its pointers catches a support that is out of order or inconsistent with
     * what it claims to hold, which materialising would hide.
     */
    fun checkSparse(
        expected: Array<DoubleArray>,
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
        check(expected, actual.toArray(), what)
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
