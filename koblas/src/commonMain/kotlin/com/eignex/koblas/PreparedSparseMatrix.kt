package com.eignex.koblas

import com.eignex.koblas.sparse.SparseAlgorithms
import com.eignex.koblas.sparse.SparseBlas
import kotlin.jvm.JvmOverloads

/**
 * An immutable snapshot of one sparse matrix, prepared for repeated products.
 *
 * The snapshot owns copies of the structure and the coefficients, so changes to the source matrix after
 * preparation do not reach it and nothing here holds a caller's mutable array. Create one with
 * [SparseMatrix.prepare].
 *
 * A prepared matrix is safe to share between concurrent readers using distinct destinations and workspaces.
 * The transposed orientation is derived on first use and published through a synchronized lazy, so a reader
 * either sees a fully built transpose or builds it; no reader can observe a half-initialized one. Mutable
 * scratch is never kept here: it belongs to the invocation, which is what keeps concurrent use safe.
 *
 * Preparation, the first transposed use and steady-state use therefore cost different things, and a
 * measurement that means to separate them has to reset between them.
 */
public class PreparedSparseMatrix internal constructor(a: SparseMatrix, private val algorithms: SparseAlgorithms) {
    private val snapshot = SparseMatrix.wrapTrusted(
        a.rows,
        a.cols,
        a.copyColumnPointers(),
        a.copyRowIndices(),
        a.values.copyOf(),
    )

    private val transposedSnapshot: SparseMatrix by lazy { algorithms.transpose(snapshot) }

    /** Rows in the prepared sparse matrix. */
    public val rows: Int get() = snapshot.rows

    /** Columns in the prepared sparse matrix. */
    public val cols: Int get() = snapshot.cols

    /** Stored entries copied into the snapshot. */
    public val nnz: Int get() = snapshot.nnz

    /** In-place `y = alpha · op(A) · x + beta · y` against the prepared `A`. */
    @Suppress("LongParameterList") // the BLAS dgemv signature
    @JvmOverloads
    public fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false) {
        algorithms.gemv(alpha, snapshot, x, beta, y, transpose)
    }

    /** Prepared selected-triangle symmetric matrix-vector product; semantics match [SparseBlas.symv]. */
    @Suppress("LongParameterList") // the BLAS dsymv signature
    @JvmOverloads
    public fun symv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean = true) {
        algorithms.symv(alpha, snapshot, x, beta, y, lower)
    }

    /** `C = alpha · op(A) · B + beta · C` against the prepared `A`. */
    @Suppress("LongParameterList") // the BLAS dgemm signature plus the workspace
    @JvmOverloads
    public fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        workspace: MatrixWorkspace? = null,
    ) {
        gemm(alpha, transposeA, b, false, beta, c, false, workspace)
    }

    /** Full sparse-dense product contract, including dense transpose and sparse side selection. */
    @Suppress("LongParameterList") // the BLAS dgemm signature, the side, and the workspace
    @JvmOverloads
    public fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        right: Boolean,
        workspace: MatrixWorkspace? = null,
    ) {
        algorithms.gemm(alpha, preparedOrientation(transposeA), false, b, transposeB, beta, c, right, workspace)
    }

    /** Prepared selected-triangle symmetric matrix-matrix product; semantics match [SparseBlas.symm]. */
    @Suppress("LongParameterList") // the BLAS dsymm signature plus the workspace
    @JvmOverloads
    public fun symm(
        alpha: Double,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
        workspace: MatrixWorkspace? = null,
    ) {
        algorithms.symm(alpha, snapshot, b, beta, c, lower, right, workspace)
    }

    /** `A · B` against the prepared `A`, into a fresh sparse matrix. */
    public fun gemm(b: SparseMatrix): SparseMatrix = gemm(1.0, false, b, false)

    /** Prepared sparse-result product with scaling and transpose controls. */
    public fun gemm(alpha: Double, transposeA: Boolean, b: SparseMatrix, transposeB: Boolean): SparseMatrix =
        algorithms.gemm(alpha, preparedOrientation(transposeA), false, b, transposeB)

    /** Prepared direct sparse-sparse-to-dense product. */
    @Suppress("LongParameterList") // the BLAS dgemm signature plus the workspace
    @JvmOverloads
    public fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: MatrixWorkspace? = null,
    ) {
        algorithms.gemm(alpha, preparedOrientation(transposeA), false, b, transposeB, beta, c, workspace)
    }

    /**
     * The snapshot in the orientation a call asks for.
     *
     * A transposed call reuses one derived transpose rather than building it per call, which is most of what
     * preparing buys for a repeated transposed product. A zero alpha still reaches it: the snapshot's values
     * are this object's own, so reading them to transpose breaks no caller's no-read contract.
     */
    private fun preparedOrientation(transpose: Boolean): SparseMatrix = if (transpose) transposedSnapshot else snapshot
}
