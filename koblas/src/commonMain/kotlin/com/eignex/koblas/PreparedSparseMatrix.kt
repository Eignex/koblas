package com.eignex.koblas

import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseCall
import com.eignex.koblas.sparse.SparseMatrixOperation
import com.eignex.koblas.sparse.SparseMatrixRoute
import com.eignex.koblas.vendor.RouteKind
import kotlin.jvm.JvmOverloads

/**
 * An immutable snapshot of one sparse matrix, prepared for repeated products with [SparseMatrix.prepare].
 * It owns copies of structure and coefficients, so source mutations do not affect it.
 *
 * As a [Matrix], it supports [Matrix.gemm] and [Matrix.gemmInto] on either side of dense, sparse or prepared
 * operands, using the engine that prepared it. Matrix-vector and selected-triangle symmetric operations
 * are also available here.
 *
 * Concurrent readers may share the snapshot with distinct destinations and workspaces. Transposed
 * sparse-sparse products lazily derive and safely publish the opposite orientation once. Dense-block and
 * matrix-vector products traverse the stored orientation directly, which measured faster than deriving
 * a transpose. Mutable scratch belongs to each invocation.
 *
 * Preparation, the first transposed sparse-sparse use and steady-state use have distinct costs;
 * measurements must reset the snapshot to separate them.
 */
public class PreparedSparseMatrix internal constructor(a: SparseMatrix, internal val blas: SparseBlas) : Matrix {
    internal val snapshot: SparseMatrix = SparseMatrix.wrapTrusted(
        a.rows,
        a.cols,
        a.copyColumnPointers(),
        a.copyRowIndices(),
        a.values.copyOf(),
    )

    private val lazyTranspose: Lazy<SparseMatrix> = lazy { blas.transpose(snapshot) }

    /**
     * Lazily cached transpose, requested only by transposed sparse-sparse products that reach a position.
     * Skipping unused transposes preserves the no-read contract and supports row counts beyond array limits.
     */
    internal val transposedSnapshot: SparseMatrix get() = lazyTranspose.value

    /** Rows in the prepared sparse matrix. */
    override val rows: Int get() = snapshot.rows

    /** Columns in the prepared sparse matrix. */
    override val cols: Int get() = snapshot.cols

    /** Stored entries copied into the snapshot. */
    public val nnz: Int get() = snapshot.nnz

    /** The snapshot's entry at row (i), column (j), or `0.0` where it stores nothing. */
    override fun get(i: Int, j: Int): Double = snapshot[i, j]

    /** Materialises the snapshot into a fresh `rows × cols` array of rows; unstored entries stay zero. */
    override fun toArray(): Array<DoubleArray> = snapshot.toArray()

    /**
     * In-place `y = alpha · op(A) · x + beta · y` against the prepared `A`.
     *
     * The transpose is taken by the operation's own flag rather than by the derived orientation, so a
     * transposed matrix-vector product costs nothing to prepare and leaves the cache alone.
     */
    @Suppress("LongParameterList") // the BLAS dgemv signature plus the workspace
    @JvmOverloads
    public fun gemvInto(
        alpha: Double,
        x: DoubleArray,
        beta: Double,
        destination: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ) {
        blas.gemv(alpha, snapshot, x, beta, destination, transpose, workspace)
    }

    /** Prepared selected-triangle symmetric matrix-vector product; semantics match [SparseBlas.symv]. */
    @Suppress("LongParameterList") // the BLAS dsymv signature plus the workspace
    @JvmOverloads
    public fun symvInto(
        alpha: Double,
        x: DoubleArray,
        beta: Double,
        destination: DoubleArray,
        lower: Boolean = true,
        workspace: Workspace? = null,
    ) {
        blas.symv(alpha, snapshot, x, beta, destination, lower, workspace)
    }

    /** Prepared selected-triangle symmetric matrix-matrix product; semantics match [SparseBlas.symm]. */
    @Suppress("LongParameterList") // the BLAS dsymm signature plus the workspace
    @JvmOverloads
    public fun symmInto(
        alpha: Double,
        b: DenseMatrix,
        beta: Double,
        destination: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
        workspace: Workspace? = null,
    ) {
        blas.symm(alpha, snapshot, b, beta, destination, lower, right, workspace)
    }

    /**
     * Reports execution against the snapshot, like [com.eignex.koblas.sparse.SparseBlas.routeOf].
     *
     * Dense-block calls traverse stored orientation. Transposed sparse-sparse calls depend on a second
     * operand absent from these facts, so their schedule is reported as unsettled.
     * [call] supplies scalars and extents; [SparseCall.matrix] is ignored in favor of this snapshot.
     */
    public fun routeOf(operation: SparseMatrixOperation, call: SparseCall): SparseMatrixRoute {
        if (orientsOnAnotherOperand(operation) && call.transposeSparse) {
            // The missing second operand decides whether any position is reached and a transpose is needed.
            return unsettled(operation, call)
        }
        return routeAgainstSnapshot(operation, call)
    }

    /** [call]'s scalars and extents asked of the snapshot, which is the operand a prepared call has. */
    private fun routeAgainstSnapshot(operation: SparseMatrixOperation, call: SparseCall): SparseMatrixRoute =
        blas.routeOf(
            operation,
            SparseCall(
                snapshot,
                call.alpha,
                call.beta,
                call.destinationElements,
                call.depth,
                call.updateRun,
                call.rightHandSides,
                call.transposeSparse,
                call.transposeDense,
                call.lower,
            ),
        )

    /** A route for a call whose schedule these facts do not settle, which names no traversal at all. */
    private fun unsettled(operation: SparseMatrixOperation, call: SparseCall): SparseMatrixRoute {
        // Preserve no-work and scaling facts; only the traversal depends on the missing operand.
        val route = routeAgainstSnapshot(operation, call)
        if (route.kind == RouteKind.NoWork) return route
        return SparseMatrixRoute(
            route.operation,
            RouteKind.Composed,
            route.scheduling,
            route.entryPoint,
            route.components.filter { it.endsWith("/scale") },
            UNDECIDED_ORIENTATION + route.reason?.let { ". Whatever runs, $it" }.orEmpty(),
            route.executionGroup,
            route.executionTail,
            resolved = false,
        )
    }

    /**
     * Whether the transposed orientation has been derived, which some transposed calls pay for once.
     *
     * Only a transposed product against a second sparse operand derives one; a product against a dense
     * block and a matrix-vector product use the stored orientation and do not initialize this cache.
     * An orientation already derived by a sparse-sparse product remains cached. A benchmark can inspect
     * this state before and after a call to identify orientation construction.
     */
    public val orientationDerived: Boolean get() = lazyTranspose.isInitialized()

    override fun toString(): String = "PreparedSparseMatrix(${rows}x$cols, nnz=$nnz)"

    /**
     * Whether this operation orients on a fact these call facts do not carry, which is the second sparse
     * operand a product against one is decided by.
     */
    private fun orientsOnAnotherOperand(operation: SparseMatrixOperation): Boolean = when (operation) {
        SparseMatrixOperation.GemmSparse, SparseMatrixOperation.GemmSparseDense -> true
        else -> false
    }
}

/** What an unsettled route says in place of the traversal a second sparse operand would have decided. */
private const val UNDECIDED_ORIENTATION: String =
    "a transposed product against a second sparse operand runs either the derived orientation or the " +
        "snapshot itself, and which one depends on that operand, which these call facts do not carry"
