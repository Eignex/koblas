@file:Suppress("VariableNaming", "FunctionParameterNaming", "TooManyFunctions") // math convention and the BLAS surface

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Vector

/**
 * A naive scalar oracle for the dense surface, defined by the textbook sums rather than by any kernel, and
 * independent of the production scheduling so that no implementation is accepted by comparison with itself.
 *
 * Every routine is the direct definition, column-major, with no blocking, no accumulator splitting and no
 * early exit beyond what the operation's contract states.
 */
internal object ReferenceBlas {
    private fun at(a: DenseMatrix, i: Int, j: Int, transpose: Boolean): Double =
        if (transpose) a.values[j + i * a.rows] else a.values[i + j * a.rows]

    /**
     * The `beta * C` term, which a zero beta contributes without reading what is there. That is not the same
     * as multiplying by zero, since `0.0 * NaN` is NaN, and tests poison their destination to check it.
     */
    private fun scaled(beta: Double, previous: Double): Double = if (beta == 0.0) 0.0 else beta * previous

    fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false) {
        val m = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        for (i in 0 until m) {
            var sum = 0.0
            for (j in 0 until k) sum += at(a, i, j, transpose) * x[j]
            y[i] = alpha * sum + scaled(beta, y[i])
        }
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature
    fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ) {
        val m = if (transposeA) a.cols else a.rows
        val depth = if (transposeA) a.rows else a.cols
        val n = if (transposeB) b.rows else b.cols
        for (j in 0 until n) {
            for (i in 0 until m) {
                var sum = 0.0
                for (p in 0 until depth) sum += at(a, i, p, transposeA) * at(b, p, j, transposeB)
                c.values[i + j * c.rows] = alpha * sum + scaled(beta, c.values[i + j * c.rows])
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS gemmt signature
    fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
    ) {
        val n = c.rows
        val depth = if (transposeA) a.rows else a.cols
        for (j in 0 until n) {
            for (i in 0 until n) {
                if (lower && i < j) continue
                if (!lower && i > j) continue
                var sum = 0.0
                for (p in 0 until depth) sum += at(a, i, p, transposeA) * at(b, p, j, transposeB)
                c.values[i + j * n] = alpha * sum + scaled(beta, c.values[i + j * n])
            }
        }
    }

    /** The symmetric operand is read from the selected triangle and mirrored, never from the other half. */
    private fun symmetric(a: DenseMatrix, i: Int, j: Int, lower: Boolean): Double =
        if (lower == (i >= j)) a.values[i + j * a.rows] else a.values[j + i * a.rows]

    @Suppress("LongParameterList") // the BLAS dsymv signature
    fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean = true) {
        val n = a.rows
        for (i in 0 until n) {
            var sum = 0.0
            for (j in 0 until n) sum += symmetric(a, i, j, lower) * x[j]
            y[i] = alpha * sum + scaled(beta, y[i])
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymm signature
    fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
    ) {
        val m = c.rows
        val n = c.cols
        for (j in 0 until n) {
            for (i in 0 until m) {
                var sum = 0.0
                if (right) {
                    for (p in 0 until n) sum += b.values[i + p * b.rows] * symmetric(a, p, j, lower)
                } else {
                    for (p in 0 until m) sum += symmetric(a, i, p, lower) * b.values[p + j * b.rows]
                }
                c.values[i + j * m] = alpha * sum + scaled(beta, c.values[i + j * m])
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature
    fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, lower: Boolean = true) {
        val n = if (transpose) a.cols else a.rows
        val depth = if (transpose) a.rows else a.cols
        for (j in 0 until n) {
            for (i in 0 until n) {
                if (lower && i < j) continue
                if (!lower && i > j) continue
                var sum = 0.0
                for (p in 0 until depth) sum += at(a, i, p, transpose) * at(a, j, p, transpose)
                c.values[i + j * n] = alpha * sum + scaled(beta, c.values[i + j * n])
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
    ) {
        val n = if (transpose) a.cols else a.rows
        val depth = if (transpose) a.rows else a.cols
        for (j in 0 until n) {
            for (i in 0 until n) {
                if (lower && i < j) continue
                if (!lower && i > j) continue
                var sum = 0.0
                for (p in 0 until depth) {
                    sum += at(a, i, p, transpose) * at(b, j, p, transpose)
                    sum += at(b, i, p, transpose) * at(a, j, p, transpose)
                }
                c.values[i + j * n] = alpha * sum + scaled(beta, c.values[i + j * n])
            }
        }
    }

    fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        for (j in 0 until a.cols) {
            for (i in 0 until a.rows) a.values[i + j * a.rows] += alpha * x[i] * y[j]
        }
    }

    fun syr(alpha: Double, x: Vector, a: DenseMatrix, lower: Boolean = true) {
        for (j in 0 until a.rows) {
            for (i in 0 until a.rows) {
                if (lower && i < j) continue
                if (!lower && i > j) continue
                a.values[i + j * a.rows] += alpha * x[i] * x[j]
            }
        }
    }

    fun syr2(alpha: Double, x: Vector, y: Vector, a: DenseMatrix, lower: Boolean = true) {
        for (j in 0 until a.rows) {
            for (i in 0 until a.rows) {
                if (lower && i < j) continue
                if (!lower && i > j) continue
                a.values[i + j * a.rows] += alpha * (x[i] * y[j] + y[i] * x[j])
            }
        }
    }

    /** The triangle entry, with an implicit unit diagonal supplied rather than read. */
    private fun triangular(a: DenseMatrix, i: Int, j: Int, unitDiag: Boolean): Double =
        if (unitDiag && i == j) 1.0 else a.values[i + j * a.rows]

    fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false) {
        val n = a.rows
        // Transposing a lower triangle gives an upper one, so the substitution order flips with it.
        val forward = lower != transpose
        val order = if (forward) 0 until n else n - 1 downTo 0
        for (i in order) {
            var sum = x[i]
            val inner = if (forward) 0 until i else i + 1 until n
            for (j in inner) sum -= entry(a, i, j, transpose, unitDiag) * x[j]
            x[i] = sum / entry(a, i, i, transpose, unitDiag)
        }
    }

    fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean = false, unitDiag: Boolean = false) {
        val n = a.rows
        val result = DoubleArray(n)
        for (i in 0 until n) {
            var sum = 0.0
            for (j in 0 until n) {
                val inTriangle = if (lower != transpose) j <= i else j >= i
                if (inTriangle) sum += entry(a, i, j, transpose, unitDiag) * x[j]
            }
            result[i] = sum
        }
        result.copyInto(x)
    }

    private fun entry(a: DenseMatrix, i: Int, j: Int, transpose: Boolean, unitDiag: Boolean): Double =
        if (transpose) triangular(a, j, i, unitDiag) else triangular(a, i, j, unitDiag)

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true)

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false)

    /**
     * Both triangular matrix routines as one pass over the right-hand sides.
     *
     * From the left the right-hand sides are the columns of B; from the right they are its rows against the
     * transposed triangle, which is the identity `X · op(A) = B` transposed into `op(A)ᵀ · Xᵀ = Bᵀ`.
     */
    @Suppress("LongParameterList") // the shared triangular signature plus the solve flag
    private fun triangularMatrix(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        solve: Boolean,
    ) {
        val sides = if (right) b.rows else b.cols
        val order = a.rows
        for (s in 0 until sides) {
            val rhs = DoubleArray(order) { k -> if (right) b.values[s + k * b.rows] else b.values[k + s * b.rows] }
            for (k in 0 until order) rhs[k] *= alpha
            val flipped = if (right) !transpose else transpose
            if (solve) trsv(a, rhs, lower, flipped, unitDiag) else trmv(a, rhs, lower, flipped, unitDiag)
            for (k in 0 until order) {
                if (right) b.values[s + k * b.rows] = rhs[k] else b.values[k + s * b.rows] = rhs[k]
            }
        }
    }
}
