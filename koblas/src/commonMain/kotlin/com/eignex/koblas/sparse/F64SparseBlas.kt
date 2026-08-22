@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64SparseMatrix

/** Sparse matrix routines as a backend half. */
public interface F64SparseBlas : Backend {
    /**
     * In-place `y = alpha · op(A) · x + beta · y`, where `op(A)` is `Aᵀ` when [transpose]. Per BLAS
     * convention `beta == 0.0` overwrites [y] without reading it, so it may arrive uninitialized.
     */
    @Suppress("LongParameterList") // the BLAS dgemv signature
    public fun gemv(
        alpha: Double,
        a: F64SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
    )

    /**
     * Solve `op(T) · x = b` in place, `op` transposing when [transpose]. [x] holds the right-hand side on
     * entry and the solution on return. Only the [lower] or upper triangle of [a] is read, and [unitDiag]
     * takes the diagonal as 1 without reading it, as the dense [com.eignex.koblas.dense.F64Blas.trsv] does.
     *
     * @throws com.eignex.koblas.SingularMatrix if a diagonal entry is missing or zero and [unitDiag] is
     *  false, naming its position.
     */
    public fun trsv(
        a: F64SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** `A · x`, or `Aᵀ · x` when [transpose], into a fresh result. */
    public fun gemv(a: F64SparseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /** [gemv] over [F64DenseVector] operands; [y] is written in place. */
    public fun gemv(
        alpha: Double,
        a: F64SparseMatrix,
        x: F64DenseVector,
        beta: Double,
        y: F64DenseVector,
        transpose: Boolean = false,
    ): Unit = gemv(alpha, a, x.data, beta, y.data, transpose)

    /** [gemv] over a [F64DenseVector], into a fresh result. */
    public fun gemv(a: F64SparseMatrix, x: F64DenseVector, transpose: Boolean = false): F64DenseVector =
        F64DenseVector.wrap(gemv(a, x.data, transpose))

    /** [trsv] over a [F64DenseVector]; [x] holds the right-hand side on entry and the solution on return. */
    public fun trsv(
        a: F64SparseMatrix,
        x: F64DenseVector,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    ): Unit = trsv(a, x.data, lower, transpose, unitDiag)
}

/**
 * The diagonal entry of column [j]. A missing entry and an explicit zero are both rejected, with different
 * messages, since a missing one is usually a construction mistake.
 */
internal fun diagonalOf(a: F64SparseMatrix, j: Int): Double {
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
