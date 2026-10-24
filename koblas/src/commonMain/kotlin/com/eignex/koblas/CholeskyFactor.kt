package com.eignex.koblas

import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Owned Cholesky factor of a symmetric positive-definite matrix, A = L Lᵀ.
 * Factorization takes O(n³) arithmetic and O(n²) storage. Solves reuse triangular BLAS;
 * this immutable factor can be shared while each caller owns its right-hand sides.
 */
public class CholeskyFactor internal constructor(private val factor: DenseMatrix) {
    /** Order of the original square matrix. */
    public val size: Int get() = factor.rows

    /** Independent lower triangular factor; its upper triangle is zero. */
    public fun lowerFactor(): DenseMatrix = DenseMatrix.wrap(size, size, factor.values.copyOf())

    /** Natural log determinant of the original matrix, computed without forming its determinant. */
    public fun logDeterminant(): Double {
        var result = 0.0
        for (i in 0 until size) result += ln(factor[i, i])
        return 2.0 * result
    }

    /** Solve A x = b, returning an independent result and leaving [right] unchanged. */
    public fun solve(right: DoubleArray): DoubleArray {
        requireShape(right.size == size) { "Cholesky solve needs $size rows, got ${right.size}" }
        return right.copyOf().also {
            koblas.trsv(factor, it, lower = true)
            koblas.trsv(factor, it, lower = true, transpose = true)
        }
    }

    /** Solve A X = B for all columns, returning independent storage. */
    public fun solve(right: DenseMatrix): DenseMatrix = solveLower(right).also {
        koblas.trsm(factor, it, lower = true, transpose = true)
    }

    /** Solve L X = B, or Lᵀ X = B when [transpose], into independent storage. */
    public fun solveLower(right: DenseMatrix, transpose: Boolean = false): DenseMatrix {
        requireShape(right.rows == size) { "Cholesky solve needs $size rows, got ${right.rows}" }
        return DenseMatrix.wrap(right.rows, right.cols, right.values.copyOf()).also {
            koblas.trsm(factor, it, lower = true, transpose = transpose)
        }
    }
}

/**
 * Factor the selected triangle of a symmetric positive-definite matrix without changing its storage.
 * Only [lower]'s triangle is read, including the diagonal; the opposite triangle may contain anything.
 * No regularization is added. A nonpositive or nonfinite pivot throws [NotPositiveDefinite] with its
 * zero-based position. Empty square matrices have an empty factor and log determinant zero.
 */
public fun DenseMatrix.cholesky(lower: Boolean = true): CholeskyFactor = factorCholesky(this, lower, koblas)

internal fun factorCholesky(input: DenseMatrix, lower: Boolean, engine: KoblasEngine): CholeskyFactor {
    requireShape(input.rows == input.cols) { "Cholesky requires a square matrix" }
    val n = input.rows
    val result = DenseMatrix.zero(n)
    for (column in 0 until n) {
        val square = engine.vectorKernels.dot(result.values, column, result.values, column, column, n, n)
        val pivot = input[column, column] - square
        if (!(pivot > 0.0 && pivot.isFinite())) {
            throw NotPositiveDefinite(column, pivot, "Cholesky pivot $column is not positive and finite: $pivot")
        }
        val diagonal = sqrt(pivot)
        result[column, column] = diagonal
        for (row in column + 1 until n) {
            val product = engine.vectorKernels.dot(result.values, row, result.values, column, column, n, n)
            result[row, column] = ((if (lower) input[row, column] else input[column, row]) - product) / diagonal
        }
    }
    return CholeskyFactor(result)
}
