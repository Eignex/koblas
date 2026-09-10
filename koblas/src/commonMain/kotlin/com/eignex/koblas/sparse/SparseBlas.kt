@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.*

/** Sparse matrix algorithms bound to immutable dense kernels. */
public interface SparseBlas {
    /**
     * Copies [a] into an immutable snapshot for repeated products.
     */
    public fun prepare(a: SparseMatrix): PreparedSparseMatrix

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
    )

    /**
     * `y = alpha · A · x + beta · y` for a square sparse [a] interpreted as symmetric. Exactly the selected
     * [lower] or upper stored triangle is read; the other triangle and implicit zeros are ignored. A stored
     * off-diagonal entry contributes to both mirrored positions. `alpha == 0.0` reads neither [a] nor [x],
     * and `beta == 0.0` overwrites [y] without reading it. Aliasing [x] with [y], or [a]'s values with [y],
     * is supported through internal staging.
     */
    @Suppress("LongParameterList")
    public fun symv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean = true,
    )

    /**
     * `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right], interpreting square
     * sparse [a] as symmetric from exactly its selected [lower] or upper triangle. `alpha == 0.0` reads
     * neither input; `beta == 0.0` does not read [c]. Shared input/destination buffers are staged, with
     * [workspace] reused when supplied.
     */
    @Suppress("LongParameterList")
    public fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
        workspace: Workspace? = null,
    )

    /**
     * Solve `op(T) · x = b` in place, `op` transposing when [transpose]. [x] holds the right-hand side on
     * entry and the solution on return. Only the [lower] or upper triangle of [a] is read, and [unitDiag]
     * takes the diagonal as 1 without reading it, as the dense [com.eignex.koblas.dense.Blas.trsv] does.
     *
     * A missing diagonal entry is the sparse representation of zero. As in dense `dtrsv`, singularity is
     * not reported: division by a zero diagonal follows IEEE 754 arithmetic when that diagonal is reached.
     */
    public fun trsv(
        a: SparseMatrix,
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
        a: SparseMatrix,
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
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
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
    public fun gemm(a: SparseMatrix, b: SparseMatrix): SparseMatrix = gemm(1.0, a, false, b, false)

    /**
     * Fresh CSC `alpha · op(A) · op(B)`. Product support is discovered from stored positions, including
     * explicit zeros, retained cancellations, and underflowed values; rows are sorted within every column
     * and all result arrays are owned. When `alpha == 0.0`, the same structure is discovered without reading
     * operand values and filled with alpha's signed zero.
     */
    @Suppress("LongParameterList")
    public fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
    ): SparseMatrix

    /**
     * Direct dense-destination `C = alpha · op(A) · op(B) + beta · C` for two sparse operands. No sparse
     * intermediate is created and implicit sparse zeros are not evaluated. `alpha == 0.0` reads neither
     * sparse operand's values; `beta == 0.0` does not read [c]. Sparse value buffers shared with [c] are
     * staged using [workspace].
     */
    @Suppress("LongParameterList")
    public fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace? = null,
    )

    /**
     * Dense selected-triangle `C = alpha · op(A) · op(A)ᵀ + beta · C`. Only the selected [lower] or upper
     * triangle of [c] is read or written, and products involving implicit sparse zeros are not evaluated.
     */
    @Suppress("LongParameterList")
    public fun syrk(
        alpha: Double,
        a: SparseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /**
     * Fresh CSC selected triangle of `op(A) · op(A)ᵀ`. Stored-product support and cancellations are
     * retained, rows are sorted, and result arrays are owned. The opposite triangle is absent rather than
     * implicitly mirrored; consume the result with [symv] or [symm] when symmetric meaning is intended.
     */
    public fun syrk(a: SparseMatrix, transpose: Boolean = false, lower: Boolean = true): SparseMatrix

    /**
     * Fresh CSC `alpha · op(A) + B`. The structural union is retained, including explicit zeros,
     * cancellations, and A-only positions when alpha is zero. Zero alpha does not read A's values; B values
     * are copied. Output rows are strictly ascending and all arrays are independently owned.
     */
    public fun addScaled(alpha: Double, a: SparseMatrix, transposeA: Boolean, b: SparseMatrix): SparseMatrix

    /**
     * `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (Sparse BLAS `ussm`).
     * The triangle flags follow [trsv]; the right-hand sides are the columns of [b] from the left and its
     * rows from the right. [workspace] reuses portable diagonal and RHS-panel staging.
     * A missing diagonal entry behaves as zero, and singularity is not reported, as in dense `dtrsm`.
     */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    public fun trsm(
        a: SparseMatrix,
        b: DenseMatrix,
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
        a: SparseMatrix,
        b: DenseMatrix,
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
    public fun transpose(a: SparseMatrix): SparseMatrix

    /** `A · x`, or `Aᵀ · x` when [transpose], into a fresh result. */
    public fun gemv(a: SparseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /** [gemm] with `alpha = 1, beta = 0`, into a fresh matrix. `A.cols` must equal `B.rows`. */
    public fun gemm(a: SparseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix.zero(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
    }
}

/**
 * An immutable snapshot of one sparse matrix, prepared for repeated products.
 *
 * Changes to the matrix passed to [SparseBlas.prepare] after preparation do not affect this snapshot.
 */
public class PreparedSparseMatrix internal constructor(a: SparseMatrix, private val algorithms: SparseAlgorithms) {
    private val snapshot = SparseMatrix.wrap(
        a.rows,
        a.cols,
        a.copyColumnPointers(),
        a.copyRowIndices(),
        a.values.copyOf(),
    )
    private var transposedSnapshot: SparseMatrix? = null

    /** Rows in the prepared sparse matrix. */
    public val rows: Int get() = snapshot.rows

    /** Columns in the prepared sparse matrix. */
    public val cols: Int get() = snapshot.cols

    /** Stored entries copied into the snapshot. */
    public val nnz: Int get() = snapshot.nnz

    /** In-place `y = alpha · op(A) · x + beta · y` against the prepared `A`. */
    @Suppress("LongParameterList")
    public fun gemv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean = false) {
        algorithms.gemv(alpha, snapshot, x, beta, y, transpose)
    }

    /** Prepared selected-triangle symmetric matrix-vector product; semantics match [SparseBlas.symv]. */
    @Suppress("LongParameterList")
    public fun symv(alpha: Double, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean = true) {
        algorithms.symv(alpha, snapshot, x, beta, y, lower)
    }

    /** `C = alpha · op(A) · B + beta · C` against the prepared `A`. [workspace] reuses portable staging on
     *  the software fallback a native binding keeps for itself. */
    @Suppress("LongParameterList")
    public fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace? = null,
    ) {
        gemm(alpha, transposeA, b, false, beta, c, false, workspace)
    }

    /** Full sparse-dense product contract, including dense transpose and sparse side selection. */
    @Suppress("LongParameterList")
    public fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        right: Boolean,
        workspace: Workspace? = null,
    ) {
        algorithms.gemm(alpha, preparedOrientation(transposeA), false, b, transposeB, beta, c, right, workspace)
    }

    /** Prepared selected-triangle symmetric matrix-matrix product; semantics match [SparseBlas.symm]. */
    @Suppress("LongParameterList")
    public fun symm(
        alpha: Double,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
        workspace: Workspace? = null,
    ) {
        algorithms.symm(alpha, snapshot, b, beta, c, lower, right, workspace)
    }

    /** `A · B` against the prepared `A`, into a fresh sparse matrix. */
    public fun gemm(b: SparseMatrix): SparseMatrix = gemm(1.0, false, b, false)

    /** Prepared sparse-result product with scaling and transpose controls. */
    public fun gemm(alpha: Double, transposeA: Boolean, b: SparseMatrix, transposeB: Boolean): SparseMatrix =
        algorithms.gemm(alpha, preparedOrientation(transposeA), false, b, transposeB)

    /** Prepared direct sparse-sparse-to-dense product. */
    @Suppress("LongParameterList")
    public fun gemm(
        alpha: Double,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace? = null,
    ) {
        algorithms.gemm(alpha, preparedOrientation(transposeA), false, b, transposeB, beta, c, workspace)
    }

    private fun preparedOrientation(transpose: Boolean): SparseMatrix {
        if (!transpose) return snapshot
        val existing = transposedSnapshot
        if (existing != null) return existing
        return algorithms.transpose(snapshot).also { transposedSnapshot = it }
    }
}
