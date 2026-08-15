@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.DenseVector
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.requireShape

/** Sparse matrix routines as a backend half. */
public interface SparseBlas : Backend {
    /**
     * In-place `y = alpha · op(A) · x + beta · y`, where `op(A)` is `Aᵀ` when [transpose]. Per BLAS
     * convention `beta == 0.0` overwrites [y] without reading it, so it may arrive uninitialized.
     */
    @Suppress("LongParameterList") // the BLAS dgemv signature
    public fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
    ) {
        val xLen = if (transpose) a.rows else a.cols
        val yLen = if (transpose) a.cols else a.rows
        requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
        requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
        when {
            beta == 0.0 -> y.fill(0.0)
            beta != 1.0 -> for (i in y.indices) y[i] *= beta
        }
        if (alpha == 0.0) return
        if (transpose) {
            for (j in 0 until a.cols) {
                var s = 0.0
                a.forEachInColumn(j) { i, v -> s += v * x[i] }
                y[j] += alpha * s
            }
        } else {
            for (j in 0 until a.cols) {
                val xj = alpha * x[j]
                if (xj != 0.0) a.forEachInColumn(j) { i, v -> y[i] += v * xj }
            }
        }
    }

    /**
     * Solve `op(T) · x = b` in place, `op` transposing when [transpose]. [x] holds the right-hand side on
     * entry and the solution on return. Only the [lower] or upper triangle of [a] is read, and [unitDiag]
     * takes the diagonal as 1 without reading it, as the dense [com.eignex.koblas.dense.Blas.trsv] does.
     *
     * @throws com.eignex.koblas.SingularMatrix if a diagonal entry is missing or zero and [unitDiag] is
     *  false, naming its position.
     */
    public fun trsv(
        a: SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    ) {
        requireShape(a.rows == a.cols) { "trsv requires a square matrix; got ${a.rows}x${a.cols}" }
        val n = a.rows
        requireShape(x.size == n) { "trsv: x length ${x.size} != $n" }
        // Forward when a finished unknown feeds later columns, backward when it feeds earlier ones.
        val order = if (lower != transpose) 0 until n else n - 1 downTo 0
        for (j in order) {
            if (!transpose) {
                val xj = if (unitDiag) x[j] else x[j] / diagonalOf(a, j)
                x[j] = xj
                if (xj != 0.0) {
                    a.forEachInColumn(j) { i, v ->
                        if (if (lower) i > j else i < j) x[i] -= v * xj
                    }
                }
            } else {
                var s = x[j]
                a.forEachInColumn(j) { i, v ->
                    if (if (lower) i > j else i < j) s -= v * x[i]
                }
                x[j] = if (unitDiag) s else s / diagonalOf(a, j)
            }
        }
    }

    /** `A · x`, or `Aᵀ · x` when [transpose], into a fresh result. */
    public fun gemv(a: SparseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /** [gemv] over [DenseVector] operands; [y] is written in place. */
    public fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        transpose: Boolean = false,
    ): Unit = gemv(alpha, a, x.data, beta, y.data, transpose)

    /** [gemv] over a [DenseVector], into a fresh result. */
    public fun gemv(a: SparseMatrix, x: DenseVector, transpose: Boolean = false): DenseVector =
        DenseVector.wrap(gemv(a, x.data, transpose))

    /** [trsv] over a [DenseVector]; [x] holds the right-hand side on entry and the solution on return. */
    public fun trsv(
        a: SparseMatrix,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    ): Unit = trsv(a, x.data, lower, transpose, unitDiag)
}

/**
 * The diagonal entry of column [j]. A missing entry and an explicit zero are both rejected, with different
 * messages, since a missing one is usually a construction mistake.
 */
private fun diagonalOf(a: SparseMatrix, j: Int): Double {
    val d = a[j, j]
    if (d == 0.0) {
        val stored = (a.colPtr[j] until a.colPtr[j + 1]).any { a.rowIdx[it] == j }
        throw SingularMatrix(
            j,
            if (stored) {
                "trsv: triangle is singular, diagonal entry $j is an explicit zero"
            } else {
                "trsv: triangle has no diagonal entry at $j, so it is singular"
            },
        )
    }
    return d
}
