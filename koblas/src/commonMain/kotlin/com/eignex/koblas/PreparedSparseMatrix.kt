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

    private val lazyTranspose: Lazy<SparseMatrix> = lazy { algorithms.transpose(snapshot) }
    private val transposedSnapshot: SparseMatrix get() = lazyTranspose.value

    /** Rows in the prepared sparse matrix. */
    public val rows: Int get() = snapshot.rows

    /** Columns in the prepared sparse matrix. */
    public val cols: Int get() = snapshot.cols

    /** Stored entries copied into the snapshot. */
    public val nnz: Int get() = snapshot.nnz

    /**
     * In-place `y = alpha · op(A) · x + beta · y` against the prepared `A`.
     *
     * The transpose is taken by the operation's own flag rather than by the derived orientation, so a
     * transposed matrix-vector product costs nothing to prepare and leaves the cache alone.
     */
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
        workspace: Workspace? = null,
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
        workspace: Workspace? = null,
    ) {
        if (right) {
            requireGemmShape(b, transposeB, snapshot, transposeA, c)
        } else {
            requireGemmShape(snapshot, transposeA, b, transposeB, c)
        }
        val depth = if (transposeA) snapshot.rows else snapshot.cols
        withOrientation(
            transposeA,
            reusesOrientation(alpha, transposeA, depth, c.values.size),
        ) { oriented, stillTransposed ->
            algorithms.gemm(alpha, oriented, stillTransposed, b, transposeB, beta, c, right, workspace)
        }
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
        workspace: Workspace? = null,
    ) {
        algorithms.symm(alpha, snapshot, b, beta, c, lower, right, workspace)
    }

    /** `A · B` against the prepared `A`, into a fresh sparse matrix. */
    public fun gemm(b: SparseMatrix): SparseMatrix = gemm(1.0, false, b, false)

    /** Prepared sparse-result product with scaling and transpose controls. */
    public fun gemm(alpha: Double, transposeA: Boolean, b: SparseMatrix, transposeB: Boolean): SparseMatrix {
        requireProductShape(transposeA, b, transposeB)
        val depth = if (transposeA) snapshot.rows else snapshot.cols
        val outputs = if (transposeB) b.rows else b.cols
        val rows = if (transposeA) snapshot.cols else snapshot.rows
        val work = if (b.nnz == 0 || rows == 0) 0 else outputs
        return withOrientation(
            transposeA,
            reusesOrientation(alpha, transposeA, depth, work),
        ) { oriented, stillTransposed ->
            algorithms.gemm(alpha, oriented, stillTransposed, b, transposeB)
        }
    }

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
        workspace: Workspace? = null,
    ) {
        requireGemmShape(snapshot, transposeA, b, transposeB, c)
        val depth = if (transposeA) snapshot.rows else snapshot.cols
        val work = if (b.nnz == 0) 0 else c.values.size
        withOrientation(transposeA, reusesOrientation(alpha, transposeA, depth, work)) { oriented, stillTransposed ->
            algorithms.gemm(alpha, oriented, stillTransposed, b, transposeB, beta, c, workspace)
        }
    }

    /**
     * Runs [block] with the snapshot in the orientation a call asks for, and the transpose flag still to apply.
     *
     * A transposed call with work to do reuses one derived transpose rather than building it per call, which
     * is most of what preparing buys for a repeated transposed product. A call without work does not: it
     * hands back the snapshot and leaves the flag set, so the operation reaches its own no-read path and the
     * numeric cache stays unbuilt. Owning the values is not a licence to read them for a product that
     * contributes nothing, and a sparse result still discovers the same structure from the pattern alone.
     * [reusesOrientation] is what decides which of the two a call is.
     */
    private inline fun <T> withOrientation(
        transpose: Boolean,
        reuse: Boolean,
        block: (SparseMatrix, Boolean) -> T,
    ): T = if (reuse) block(transposedSnapshot, false) else block(snapshot, transpose)

    /**
     * Whether a transposed call should derive and reuse the snapshot's transpose.
     *
     * Only a call that will traverse the operand. A zero multiplier reads no values, an empty destination or
     * an empty inner extent computes nothing, and a snapshot with nothing stored reaches no position: in each
     * case the operation has its own path that never looks at the coefficients, and building a transpose to
     * reach it would both break the no-read contract and, for a snapshot with more rows than an array can
     * index, turn a valid empty product into a shape error.
     */
    private fun reusesOrientation(alpha: Double, transpose: Boolean, depth: Int, outputs: Int): Boolean =
        transpose && alpha != 0.0 && depth > 0 && outputs > 0 && snapshot.nnz > 0

    /** The shape a fresh sparse product needs, checked before any orientation is derived. */
    private fun requireProductShape(transposeA: Boolean, b: SparseMatrix, transposeB: Boolean) {
        val aRows = if (transposeA) snapshot.cols else snapshot.rows
        val aCols = if (transposeA) snapshot.rows else snapshot.cols
        val bRows = if (transposeB) b.cols else b.rows
        val bCols = if (transposeB) b.rows else b.cols
        requireShape(aCols == bRows) { "gemm: op(A) is ${aRows}x$aCols but op(B) is ${bRows}x$bCols" }
    }

    /** Whether the derived transposed orientation has been built, for tests that assert it is not. */
    internal val transposeDerived: Boolean get() = lazyTranspose.isInitialized()
}
