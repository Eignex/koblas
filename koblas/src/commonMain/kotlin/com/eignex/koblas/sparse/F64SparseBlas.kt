@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SingularMatrix
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
     * Unlike the dense half, which follows `dtrsv` in reporting nothing and answering a singular triangle
     * with infinities, this reports it. There is no sparse BLAS routine whose silence it has to match, and
     * the value is looked up to divide by it anyway, so a healthy solve pays nothing; only the failing
     * column pays the extra scan that tells a stored zero from a missing entry.
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
