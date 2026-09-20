package com.eignex.koblas

import com.eignex.koblas.sparse.SparseAlgorithms
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseCall
import com.eignex.koblas.sparse.SparseMatrixOperation
import com.eignex.koblas.sparse.SparseMatrixRoute
import com.eignex.koblas.vendor.RouteKind
import kotlin.jvm.JvmOverloads

/**
 * An immutable snapshot of one sparse matrix, prepared for repeated products.
 *
 * The snapshot owns copies of the structure and the coefficients, so changes to the source matrix after
 * preparation do not reach it and nothing here holds a caller's mutable array. Create one with
 * [SparseMatrix.prepare].
 *
 * A prepared matrix is safe to share between concurrent readers using distinct destinations and workspaces.
 * A transposed product against a second sparse operand derives the opposite orientation on first use and
 * publishes it through a synchronized lazy, so a reader either sees a fully built transpose or builds it;
 * no reader can observe a half-initialized one. The products against a dense block and the matrix-vector
 * products take their transpose flag straight to the operation and derive nothing, because the transposed
 * traversal over the stored orientation measured faster than the untransposed one over a derived transpose.
 * Mutable scratch is never kept here: it belongs to the invocation, which is what keeps concurrent use safe.
 *
 * Preparation, the first transposed use of a sparse-sparse product and steady-state use therefore cost
 * different things, and a measurement that means to separate them has to reset between them.
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

    /**
     * Full sparse-dense product contract, including dense transpose and sparse side selection.
     *
     * Uses the stored snapshot and the original transpose flags on either side, without deriving another
     * orientation. Deriving a transpose changes the CSC traversal as well as adding preparation work;
     * dense products retain the same traversal as a one-shot call. Products against a second sparse
     * operand have a separate policy and may reuse the cached transpose.
     */
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
        algorithms.gemm(alpha, snapshot, transposeA, b, transposeB, beta, c, right, workspace)
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
     * Only the products against a second sparse operand reach this. A transposed one with work to do reuses
     * one derived transpose rather than building it per call, which is what preparing buys them: the
     * calibration measured that schedule ahead of the transposed traversal for a sparse result and behind it
     * for a dense one, so the two families differ here and the dense products pass their flag straight
     * through. A call without work does not orient either: it hands back the snapshot and leaves the flag
     * set, so the operation reaches its own no-read path and the numeric cache stays unbuilt. Owning the
     * values is not a licence to read them for a product that contributes nothing, and a sparse result still
     * discovers the same structure from the pattern alone. [reusesOrientation] is what decides which of the
     * two a call is.
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

    /**
     * What a repeated call of this shape against the snapshot executes.
     *
     * The counterpart of [com.eignex.koblas.sparse.SparseBlas.matrixRouteOf] for a prepared operand. For
     * every product against a dense block it is the same answer, because such a call runs the schedule its
     * own flags ask for over the stored orientation, exactly as a one-shot call does. What differs is a
     * product against a second sparse operand, which may run the derived orientation instead; that operand
     * is not among these facts, so the answer says the schedule is unsettled rather than guessing it.
     *
     * [call]'s scalars and extents describe the call; its [SparseCall.matrix] is not read, because the
     * operand is this snapshot.
     */
    public fun matrixRouteOf(operation: SparseMatrixOperation, call: SparseCall): SparseMatrixRoute {
        if (orientsOnAnotherOperand(operation) && call.transposeSparse) {
            // A product against a second sparse operand decides on that operand: nothing is oriented for a
            // call that reaches no position. These facts carry no second operand, so which of the two
            // schedules runs is not derivable from them, and deriving an orientation to answer would both
            // guess and pay for the guess.
            return unsettled(operation, call)
        }
        return routeAgainstSnapshot(operation, call)
    }

    /** [call]'s scalars and extents asked of the snapshot, which is the operand a prepared call has. */
    private fun routeAgainstSnapshot(operation: SparseMatrixOperation, call: SparseCall): SparseMatrixRoute =
        algorithms.matrixRouteOf(
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
        // The call's own facts first, against the snapshot as it stands. What they settle stays: a call
        // with nothing to do is still a call with nothing to do whichever orientation it would have used,
        // and a destination multiplier still scales a destination. What they do not settle is which of the
        // two traversals runs, and only that is replaced by saying so.
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

    /**
     * Whether this operation orients on a fact these call facts do not carry, which is the second sparse
     * operand a product against one is decided by.
     */
    private fun orientsOnAnotherOperand(operation: SparseMatrixOperation): Boolean = when (operation) {
        SparseMatrixOperation.GemmSparse, SparseMatrixOperation.GemmSparseDense -> true
        else -> false
    }

    /** The shape a fresh sparse product needs, checked before any orientation is derived. */
    private fun requireProductShape(transposeA: Boolean, b: SparseMatrix, transposeB: Boolean) {
        val aRows = if (transposeA) snapshot.cols else snapshot.rows
        val aCols = if (transposeA) snapshot.rows else snapshot.cols
        val bRows = if (transposeB) b.cols else b.rows
        val bCols = if (transposeB) b.rows else b.cols
        requireShape(aCols == bRows) { "gemm: op(A) is ${aRows}x$aCols but op(B) is ${bRows}x$bCols" }
    }
}

/** What an unsettled route says in place of the traversal a second sparse operand would have decided. */
private const val UNDECIDED_ORIENTATION: String =
    "a transposed product against a second sparse operand runs either the derived orientation or the " +
        "snapshot itself, and which one depends on that operand, which these call facts do not carry"
