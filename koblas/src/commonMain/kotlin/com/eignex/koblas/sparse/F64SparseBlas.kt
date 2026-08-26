@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.core.F64DenseMatrix
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

    /**
     * `C = alpha · op(A) · op(B) + beta · C` (Sparse BLAS `usmm`) for a sparse [a] against a dense [b] and
     * [c], or `C = alpha · op(B) · op(A) + beta · C` when [right]. `beta == 0.0` overwrites [c] without
     * reading it, as [gemv] does.
     *
     * A sparse matrix times a dense one stays dense and fills in nowhere, which is what puts this on the
     * sparse half at all: it is [gemv] over the columns of the dense operand.
     *
     * [right] is here for the reason `dtrsm` has a side and `dgemm` does not. `dgemm` needs none because its
     * operands are the same kind and a caller picks which is which; here [a] is distinguished by being the
     * sparse one and cannot be swapped, so without a side the product of a dense matrix by a sparse one is
     * only reachable by transposing the result. A host binding may well accelerate one side and not the
     * other, since the routines these libraries carry take the sparse operand on the left.
     */
    @Suppress("LongParameterList") // the BLAS dgemm signature, plus the side the sparse operand sits on
    public fun gemm(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        right: Boolean = false,
    )

    /**
     * `C = A · B` for two sparse operands, into a fresh matrix.
     *
     * The one routine here that returns its result rather than filling a destination, because the result's
     * pattern is what the multiplication discovers: there is no destination to hand in before it is known,
     * and no meaningful `beta · C` to accumulate into. [transpose] is the same shape and returns for the
     * same reason.
     *
     * `A.cols` must equal `B.rows`. An entry the arithmetic cancels to zero is kept, as a structural
     * operation keeps one.
     */
    public fun gemm(a: F64SparseMatrix, b: F64SparseMatrix): F64SparseMatrix

    /**
     * `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (Sparse BLAS `ussm`).
     * The triangle flags follow [trsv]; the right-hand sides are the columns of [b] from the left and its
     * rows from the right.
     *
     * @throws com.eignex.koblas.SingularMatrix as [trsv] does, from the first right-hand side to reach the
     *  offending diagonal entry, so [b] may be partly solved when it is raised.
     */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    public fun trsm(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
    )

    /**
     * Fresh transposed [a], still CSC, which makes this the CSC-to-CSR conversion as well. Explicitly stored
     * zeros survive, since the transpose is structural rather than arithmetic.
     *
     * On the seam rather than beside the other whole-matrix operations because it is the one of them a
     * library has its own routine for, and because the two directions of [gemv] leave a caller with no
     * reason to materialize a transpose unless it means to hold it.
     */
    public fun transpose(a: F64SparseMatrix): F64SparseMatrix

    /** `A · x`, or `Aᵀ · x` when [transpose], into a fresh result. */
    public fun gemv(a: F64SparseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /** [gemm] with `alpha = 1, beta = 0`, into a fresh matrix. `A.cols` must equal `B.rows`. */
    public fun gemm(a: F64SparseMatrix, b: F64DenseMatrix): F64DenseMatrix {
        val c = F64DenseMatrix.zero(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
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
