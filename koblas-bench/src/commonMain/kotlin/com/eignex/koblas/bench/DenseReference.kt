@file:Suppress("VariableNaming", "FunctionParameterNaming", "TooManyFunctions") // math and the dense surface

package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import kotlin.math.abs

/**
 * An explicit scalar reference for the dense cases, used to check a case before it is timed.
 *
 * Nothing here shares code with the library's dense scheduling, its panels or its kernels. Every routine is
 * the textbook definition read through the public [DenseMatrix] accessor, with no blocking, no grouping and
 * no early exit beyond what the operation's contract states. A benchmark publishing a number has to have
 * computed the right thing, and a case whose result disagrees is a failure rather than a fast row.
 *
 * The fixtures are finite, so a tolerance is the right comparison; the library's own conformance tests are
 * where exceptional values are pinned to hand-computed answers.
 */
internal object DenseReference {
    /** Relative tolerance for the same terms summed in a different order, with or without a fused multiply. */
    private const val TOLERANCE = 1e-9

    private fun at(a: DenseMatrix, i: Int, j: Int, transpose: Boolean): Double = if (transpose) a[j, i] else a[i, j]

    private fun symmetricAt(a: DenseMatrix, i: Int, j: Int, lower: Boolean): Double =
        if (lower == (i >= j)) a[i, j] else a[j, i]

    private fun triangularAt(a: DenseMatrix, i: Int, j: Int, transpose: Boolean, unitDiag: Boolean): Double = when {
        unitDiag && i == j -> 1.0
        transpose -> a[j, i]
        else -> a[i, j]
    }

    private fun scaled(beta: Double, previous: Double): Double = if (beta == 0.0) 0.0 else beta * previous

    /** `y = alpha·op(A)·x + beta·y`. */
    fun gemv(alpha: Double, a: DenseMatrix, transpose: Boolean, x: DoubleArray, beta: Double, y: DoubleArray) =
        DoubleArray(if (transpose) a.cols else a.rows) { i ->
            var sum = 0.0
            for (p in 0 until (if (transpose) a.rows else a.cols)) sum += at(a, i, p, transpose) * x[p]
            alpha * sum + scaled(beta, y[i])
        }

    /** `y = alpha·A·x + beta·y` reading only the selected triangle of a symmetric `A`. */
    fun symv(alpha: Double, a: DenseMatrix, lower: Boolean, x: DoubleArray, beta: Double, y: DoubleArray) =
        DoubleArray(a.rows) { i ->
            var sum = 0.0
            for (j in 0 until a.cols) sum += symmetricAt(a, i, j, lower) * x[j]
            alpha * sum + scaled(beta, y[i])
        }

    /** `A += alpha·x·yᵀ`. */
    fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix): DoubleArray =
        DoubleArray(a.rows * a.cols) { at -> a.values[at] + alpha * x[at % a.rows] * y[at / a.rows] }

    /** `A += alpha·x·xᵀ` in the selected triangle, leaving the other one as it was. */
    fun syr(alpha: Double, x: DoubleArray, a: DenseMatrix, lower: Boolean): DoubleArray =
        DoubleArray(a.rows * a.cols) { at ->
            val i = at % a.rows
            val j = at / a.rows
            if (if (lower) i >= j else i <= j) a.values[at] + alpha * x[i] * x[j] else a.values[at]
        }

    /** `A += alpha·(x·yᵀ + y·xᵀ)` in the selected triangle. */
    fun syr2(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix, lower: Boolean): DoubleArray =
        DoubleArray(a.rows * a.cols) { at ->
            val i = at % a.rows
            val j = at / a.rows
            if (if (lower) i >= j else i <= j) {
                a.values[at] + alpha * (x[i] * y[j] + y[i] * x[j])
            } else {
                a.values[at]
            }
        }

    /** `x = op(T)·x` over the selected triangle. */
    fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean): DoubleArray {
        val n = a.rows
        return DoubleArray(n) { i ->
            var sum = 0.0
            for (j in 0 until n) {
                val inTriangle = if (lower != transpose) j <= i else j >= i
                if (inTriangle) sum += triangularAt(a, i, j, transpose, unitDiag) * x[j]
            }
            sum
        }
    }

    /** `op(T)·x = b` solved by substitution over the selected triangle. */
    fun trsv(a: DenseMatrix, b: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean): DoubleArray {
        val n = a.rows
        val x = b.copyOf()
        val forward = lower != transpose
        for (i in if (forward) 0 until n else n - 1 downTo 0) {
            var sum = x[i]
            for (j in if (forward) 0 until i else i + 1 until n) {
                sum -= triangularAt(a, i, j, transpose, unitDiag) * x[j]
            }
            x[i] = sum / triangularAt(a, i, i, transpose, unitDiag)
        }
        return x
    }

    /** `C = alpha·op(A)·op(B) + beta·C`. */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ): DoubleArray {
        val depth = if (transposeA) a.rows else a.cols
        return DoubleArray(c.rows * c.cols) { index ->
            val i = index % c.rows
            val j = index / c.rows
            var sum = 0.0
            for (p in 0 until depth) sum += at(a, i, p, transposeA) * at(b, p, j, transposeB)
            alpha * sum + scaled(beta, c.values[index])
        }
    }

    /** [gemm] written into one triangle of a square destination, leaving the other one as it was. */
    @Suppress("LongParameterList") // the BLAS gemmt signature
    fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
    ): DoubleArray {
        val full = gemm(alpha, a, transposeA, b, transposeB, beta, c)
        return DoubleArray(c.rows * c.cols) { index ->
            val i = index % c.rows
            val j = index / c.rows
            if (if (lower) i >= j else i <= j) full[index] else c.values[index]
        }
    }

    /** `C = alpha·A·B + beta·C`, or with the symmetric operand on the right. */
    @Suppress("LongParameterList") // the BLAS dsymm signature
    fun symm(
        alpha: Double,
        a: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
    ): DoubleArray = DoubleArray(c.rows * c.cols) { index ->
        val i = index % c.rows
        val j = index / c.rows
        var sum = 0.0
        for (p in 0 until (if (right) c.cols else c.rows)) {
            sum += if (right) b[i, p] * symmetricAt(a, p, j, lower) else symmetricAt(a, i, p, lower) * b[p, j]
        }
        alpha * sum + scaled(beta, c.values[index])
    }

    /** `C = alpha·op(A)·op(A)ᵀ + beta·C`, or its two-operand form, in the selected triangle. */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    fun syrk(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        doubled: Boolean,
    ): DoubleArray {
        val depth = if (transpose) a.rows else a.cols
        return DoubleArray(c.rows * c.cols) { index ->
            val i = index % c.rows
            val j = index / c.rows
            if (if (lower) i < j else i > j) {
                c.values[index]
            } else {
                var sum = 0.0
                for (p in 0 until depth) {
                    val aip = at(a, i, p, transpose)
                    val ajp = at(a, j, p, transpose)
                    sum += if (doubled) aip * at(b, j, p, transpose) + at(b, i, p, transpose) * ajp else aip * ajp
                }
                alpha * sum + scaled(beta, c.values[index])
            }
        }
    }

    /** `B = alpha·op(T)·B`, or `B = alpha·B·op(T)` from the right. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ): DoubleArray = triangularBlock(a, b, lower, transpose, unitDiag, right, alpha, solve = false)

    /** `B = alpha·op(T)⁻¹·B`, or `B = alpha·B·op(T)⁻¹` from the right. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ): DoubleArray = triangularBlock(a, b, lower, transpose, unitDiag, right, alpha, solve = true)

    @Suppress("LongParameterList") // the BLAS dtrsm signature plus which of the two it is
    private fun triangularBlock(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        solve: Boolean,
    ): DoubleArray {
        val order = a.rows
        val result = DoubleArray(b.rows * b.cols)
        for (s in 0 until (if (right) b.rows else b.cols)) {
            val rhs = DoubleArray(order) { k -> alpha * if (right) b[s, k] else b[k, s] }
            val flipped = if (right) !transpose else transpose
            val done = if (solve) trsv(a, rhs, lower, flipped, unitDiag) else trmv(a, rhs, lower, flipped, unitDiag)
            for (k in 0 until order) {
                result[if (right) s + k * b.rows else k + s * b.rows] = done[k]
            }
        }
        return result
    }


    /** `y(c) = alpha·Σ a(i, c)·x(i) + beta·y(c)`, the multi-dot panel written out. */
    fun multiDot(alpha: Double, a: DoubleArray, rows: Int, columns: Int, x: DoubleArray, beta: Double, y: DoubleArray) =
        DoubleArray(columns) { c ->
            var sum = 0.0
            for (i in 0 until rows) sum += a[i + c * rows] * x[i]
            alpha * sum + scaled(beta, y[c])
        }

    /** `y(i) += Σ (alpha·x(c))·a(i, c)`, the column-update panel written out. */
    fun columnUpdate(alpha: Double, a: DoubleArray, rows: Int, columns: Int, x: DoubleArray, y: DoubleArray) =
        DoubleArray(rows) { i ->
            var sum = y[i]
            for (c in 0 until columns) sum += (alpha * x[c]) * a[i + c * rows]
            sum
        }

    /** The coupled panel written out, as the updated window followed by the accumulated sums. */
    @Suppress("LongParameterList") // the panel window, both vectors and both destinations
    fun coupledUpdateDot(
        alpha: Double,
        a: DoubleArray,
        rows: Int,
        columns: Int,
        x: DoubleArray,
        coefficients: DoubleArray,
        y: DoubleArray,
        sums: DoubleArray,
    ): DoubleArray {
        val updated = columnUpdate(alpha, a, rows, columns, coefficients, y)
        val reduced = DoubleArray(columns) { c ->
            var sum = 0.0
            for (i in 0 until rows) sum += a[i + c * rows] * x[i]
            sums[c] + alpha * sum
        }
        return updated + reduced
    }

    /** `a(i, c) += (alpha·coefficients(c))·x(i)`, the rank-update panel written out. */
    fun rankUpdate(
        alpha: Double,
        a: DoubleArray,
        rows: Int,
        columns: Int,
        x: DoubleArray,
        coefficients: DoubleArray,
    ) = DoubleArray(rows * columns) { at -> a[at] + (alpha * coefficients[at / rows]) * x[at % rows] }

    /** Fails when [actual] differs from [expected] anywhere in the whole buffer. */
    fun check(expected: DoubleArray, actual: DoubleArray, what: String) {
        check(expected.size == actual.size) { "$what: ${actual.size} entries, expected ${expected.size}" }
        for (i in expected.indices) {
            val bound = TOLERANCE * maxOf(1.0, abs(expected[i]))
            check(abs(expected[i] - actual[i]) <= bound) { "$what at $i: ${actual[i]}, expected ${expected[i]}" }
        }
    }
}
