@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.Backend
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.dense.*

/** Sparse matrix routines as a backend half. */
public interface F64SparseBlas : Backend {
    /**
     * Prepares an immutable snapshot of [a] for repeated products. The caller owns the returned resource and
     * should close it with `use`. Portable backends retain an ordinary CSC copy; native backends may retain
     * native descriptors and buffers.
     */
    public fun prepare(a: F64SparseMatrix): F64PreparedSparseMatrix = ReferencePreparedSparseMatrix(a)

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
     * A missing diagonal entry is the sparse representation of zero. As in dense `dtrsv`, singularity is
     * not reported: division by a zero diagonal follows IEEE 754 arithmetic when that diagonal is reached.
     */
    public fun trsv(
        a: F64SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /**
     * Multiply `x = op(T) · x` in place (BLAS `dtrmv`), where `op` transposes when [transpose]. Only the
     * [lower] or upper triangle is read; [unitDiag] treats its diagonal as one without reading it.
     *
     * Unlike [trsv], this performs no division and therefore has no singularity behavior: a missing diagonal
     * is simply zero. The destination may share [a]'s value buffer.
     */
    public fun trmv(
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
     * other, since the routines these libraries carry take the sparse operand on the left. [workspace] reuses
     * portable dense-panel and transpose staging.
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
        workspace: Workspace? = null,
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
     * rows from the right. [workspace] reuses portable diagonal and RHS-panel staging.
     * A missing diagonal entry behaves as zero, and singularity is not reported, as in dense `dtrsm`.
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
        workspace: Workspace? = null,
    )

    /**
     * `B = alpha · op(T) · B` in place, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`). The triangle
     * flags follow [trmv]. This is multiplication rather than a solve, so zero or missing diagonal entries
     * participate normally and are never treated as singular. The destination may share [a]'s value buffer.
     */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    public fun trmm(
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
 * An explicitly owned immutable snapshot of one sparse matrix, prepared for repeated products.
 *
 * Changes to the matrix passed to [F64SparseBlas.prepare] after preparation do not affect this snapshot.
 * Calls after [close] throw [IllegalStateException], and close is idempotent. A handle and close must not be
 * used concurrently; callers that share a handle between threads must serialize its operations.
 */
public interface F64PreparedSparseMatrix : AutoCloseable {
    /** Rows in the prepared sparse matrix. */
    public val rows: Int

    /** Columns in the prepared sparse matrix. */
    public val cols: Int

    /** Stored entries copied into the snapshot. */
    public val nnz: Int

    /** In-place `y = alpha · op(A) · x + beta · y` against the prepared `A`. */
    @Suppress("LongParameterList")
    public fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false)

    /** `C = alpha · op(A) · B + beta · C` against the prepared `A`. [workspace] reuses portable staging on
     *  the software fallback a native binding keeps for itself. */
    @Suppress("LongParameterList")
    public fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        workspace: Workspace? = null,
    )

    /** `A · B` against the prepared `A`, into a fresh sparse matrix. */
    public fun gemm(b: F64SparseMatrix): F64SparseMatrix

    /** Releases resources owned by this prepared snapshot. */
    override fun close()
}

private class ReferencePreparedSparseMatrix(a: F64SparseMatrix) : F64PreparedSparseMatrix {
    private val snapshot = sparseSnapshotOf(a)
    private var closed = false

    override val rows: Int get() = snapshot.rows
    override val cols: Int get() = snapshot.cols
    override val nnz: Int get() = snapshot.nnz

    override fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        checkOpen()
        F64ReferenceSparseLinearAlgebra.gemv(alpha, snapshot, x, beta, y, transpose)
    }

    override fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        workspace: Workspace?,
    ) {
        checkOpen()
        F64ReferenceSparseLinearAlgebra.gemm(alpha, snapshot, transposeA, b, false, beta, c, workspace = workspace)
    }

    override fun gemm(b: F64SparseMatrix): F64SparseMatrix {
        checkOpen()
        return F64ReferenceSparseLinearAlgebra.gemm(snapshot, b)
    }

    override fun close() {
        closed = true
    }

    private fun checkOpen() {
        check(!closed) { "prepared sparse matrix is closed" }
    }
}

/** Copies a validated CSC matrix so later mutation through its expert escape hatches cannot stale a handle. */
internal fun sparseSnapshotOf(a: F64SparseMatrix): F64SparseMatrix = F64SparseMatrix.wrap(
    a.rows,
    a.cols,
    a.copyColumnPointers(),
    a.copyRowIndices(),
    a.values.copyOf(),
)
