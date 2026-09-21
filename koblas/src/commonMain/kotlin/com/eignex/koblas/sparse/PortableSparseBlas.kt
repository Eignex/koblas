@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.dense.scaleComponent
import com.eignex.koblas.sparse.internal.SparseAccumulationKernels
import com.eignex.koblas.sparse.internal.forEachPanelWidth
import com.eignex.koblas.sparse.internal.forEachTriangleColumn
import com.eignex.koblas.sparse.internal.multiplyFromTheLeft
import com.eignex.koblas.sparse.internal.multiplyFromTheRight
import com.eignex.koblas.sparse.internal.multiplySparse
import com.eignex.koblas.sparse.internal.multiplySparseInto
import com.eignex.koblas.sparse.internal.multiplySymmetricFromTheLeft
import com.eignex.koblas.sparse.internal.pointerLength
import com.eignex.koblas.sparse.internal.productRhsPlan
import com.eignex.koblas.sparse.internal.rhsStaged
import com.eignex.koblas.sparse.internal.rhsWidth
import com.eignex.koblas.sparse.internal.symmetricRankInto
import com.eignex.koblas.sparse.internal.symmetricRankProduct
import com.eignex.koblas.sparse.internal.symmetricRhsPlan
import com.eignex.koblas.sparse.internal.transposeCsc
import com.eignex.koblas.sparse.internal.triangularForward
import com.eignex.koblas.sparse.internal.triangularLeftCore
import com.eignex.koblas.sparse.internal.triangularRhsPlan
import com.eignex.koblas.sparse.internal.trmmRightCore
import com.eignex.koblas.sparse.internal.trmvCore
import com.eignex.koblas.sparse.internal.trsmRightCore
import com.eignex.koblas.sparse.internal.trsvCore
import com.eignex.koblas.sparse.internal.withExplicitDiagonal
import com.eignex.koblas.sparse.internal.withSymmetricRankScratch
import com.eignex.koblas.vendor.RouteKind

/** The component name every built-in sparse Level 2 and 3 call reports, whatever Level 1 kernels it calls. */
internal const val SPARSE_SCHEDULING: String = "portable-csc"

/** The component a call names where it copies a panel of right-hand sides into adjacent order and back. */
internal const val SPARSE_STAGING: String = "portable-stage"

/** The leaf name for a panel that reduces the rows a column stores into one output row. */
private const val GATHER_PANEL: String = "sparse-rhs-gather"

/** The leaf name for a panel that spreads one index across the rows a column stores. */
private const val SCATTER_PANEL: String = "sparse-rhs-scatter"

/** The leaf name for the two panels a symmetric column reaches, which are one of each. */
private const val SYMMETRIC_PANEL: String = "sparse-rhs-mirror"

/**
 * Which runs of a sparse column a call hands to a panel, and what a staged call copies.
 *
 * The two are one description because they are the same question asked of one operation: what the panels
 * are given decides which bodies run, and what has to be copied for them to be adjacent decides what the
 * staging costs. A source is read into adjacent order and never written back; a destination is read in and
 * written back, and a symmetric product does both because its two dense operands are different arrays.
 *
 * @property shape which run of a column one panel call is given, which differs by family: a general
 *   product hands over every stored entry, a symmetric one hands over the selected part in a single
 *   coupled pass with the diagonal in it, and a triangular one hands over the strictly triangular part.
 * @property stagedComponents what a staged call copies, named as the route names it.
 * @param reduces whether this call's panels carry an accumulator per right-hand side rather than a window
 *   of a destination, which is the shape the backend is asked about and may group differently. A triangular
 *   routine answers with its own transpose flag instead, through `reduction`, since the direction that
 *   gathers finished rows into a pivot reduces and the direction that spreads a pivot does not.
 * @param solve which direction a triangular dependence runs, or null where the operation has none.
 */
private enum class PanelRuns(
    val shape: RunShape,
    val stagedComponents: List<String>,
    private val reduces: Boolean = false,
    private val solve: Boolean? = null,
) {
    /** A product whose dense operand is the source that is staged, and whose destination is an accumulator. */
    WholeColumnSource(RunShape.WholeColumn, listOf("rhs-source"), reduces = true),

    /** A product whose destination is the dense block, which is staged and written back. */
    WholeColumnDestination(RunShape.WholeColumn, listOf("rhs-destination")),

    /** A symmetric product, whose selected run is scattered and mirrored back in one coupled pass. */
    Symmetric(RunShape.SelectedRun, listOf("rhs-source", "rhs-destination")),

    /** A triangular solve, whose panels are the strictly triangular part of a column. */
    TriangularSolve(RunShape.StrictRun, listOf("rhs-destination"), solve = true),

    /** A triangular multiply, which is the same panels in the other dependence direction. */
    TriangularMultiply(RunShape.StrictRun, listOf("rhs-destination"), solve = false),
    ;

    /** How a staged copy is described in a route's reason, which is what it copies and which way. */
    val stagedReason: String
        get() {
            val copy = when {
                stagedComponents.size > 1 ->
                    "both dense blocks are copied into adjacent right-hand sides and the destination is " +
                        "written back"

                stagedComponents.single() == "rhs-source" ->
                    "the dense operand is copied into adjacent right-hand sides and nothing is written " +
                        "back through it"

                else -> "the destination is copied into adjacent right-hand sides and written back"
            }
            return "$copy, which is what lets a panel be a vector body; "
        }

    /**
     * The leaf a panel of this call reports, which is the direction it runs in.
     *
     * A symmetric column is one coupled pass that both gathers and scatters, so it has a leaf of its own;
     * every other family is one of the two.
     */
    fun leaf(transpose: Boolean): String = when {
        this == Symmetric -> SYMMETRIC_PANEL
        reduction(transpose) -> GATHER_PANEL
        else -> SCATTER_PANEL
    }

    /**
     * Whether a group keeps the written-out loop where one of its right-hand sides is not live.
     *
     * Only a triangular routine that spreads a pivot does: it skips a right-hand side whose pivot is zero,
     * which a panel over the group cannot, so such a group runs the loop instead of the panel.
     */
    fun masked(transpose: Boolean): Boolean = solve != null && !transpose

    /**
     * Whether the columns are visited from the first, which only a triangular routine answers with anything
     * but yes.
     *
     * The same expression the triangular scheduling takes its direction from, because the route has to list
     * the bodies in the order the traversal reached them and a dependence decides that order.
     */
    fun forward(lower: Boolean, transpose: Boolean): Boolean =
        solve?.let { triangularForward(lower, transpose, it) } ?: true

    /**
     * The plan this call's right-hand sides are grouped and staged by, which is the one the call makes.
     *
     * Asked through the family's own function rather than restated here, so a route cannot name an adjacent
     * panel for a call that will run a strided one or a width the call did not take.
     */
    fun plan(kernels: SparsePanelKernels, call: SparseCall): Int {
        val a = call.matrix
        val sides = call.rightHandSides
        return when (shape) {
            RunShape.WholeColumn -> productRhsPlan(
                kernels,
                if (call.transposeSparse) a.cols else a.rows,
                sides,
                if (call.transposeSparse) a.rows else a.cols,
                a.nnz,
                call.transposeSparse,
                call.transposeDense,
            )

            RunShape.SelectedRun -> symmetricRhsPlan(kernels, a.rows, sides, a.nnz)

            RunShape.StrictRun -> triangularRhsPlan(kernels, a.rows, sides, a.nnz, call.transposeSparse)
        }
    }

    /**
     * Whether this call's right-hand sides are adjacent where they lie, so a panel reaches a vector body
     * without a staged copy. Only a reduction whose dense operand is already in that order is.
     */
    fun nativelyContiguous(call: SparseCall): Boolean =
        shape == RunShape.WholeColumn && call.transposeSparse && call.transposeDense

    /**
     * Which of the two sparse panel shapes this call's leaves are, which a triangular routine answers with
     * its own transpose flag.
     *
     * The gathering direction reduces finished rows into a pivot through the reduction leaf; the scattering
     * direction spreads a pivot through the update one. Asking the backend about the shape that does not
     * run would take a width chosen for the other body and report that body's name.
     */
    fun reduction(transpose: Boolean): Boolean = if (solve == null) reduces else transpose
}

/**
 * Which run of a sparse column one panel call is handed.
 *
 * One call, one run, including the symmetric column: its two halves are one coupled pass over the selected
 * run, so a route that enumerated both would name a body for a call that never happens.
 */
private enum class RunShape {
    /** Every stored entry of the column. */
    WholeColumn,

    /** The entries on the selected side of the diagonal, the diagonal included. */
    SelectedRun,

    /** The entries strictly off the diagonal on the selected side. */
    StrictRun,
}

/**
 * Portable CSC matrix algorithms, bound to one engine's dense and indexed Level 1 kernels.
 *
 * Traversal, structure, alias staging, triangle selection and arithmetic order are written here and are the
 * same on every platform. The kernels are reached only where a unit of work is a contiguous run or an indexed
 * slice; [routeOf] is what says which of the two a given call is.
 */
@Suppress("TooManyFunctions") // the sparse BLAS surface
@OptIn(UnsafeKoblasApi::class)
internal class PortableSparseBlas(
    private val vectorKernels: DenseVectorKernels,
    private val indexedKernels: IndexedSparseKernels,
    private val panelKernels: SparsePanelKernels,
) : SparseBlas {
    override fun prepare(a: SparseMatrix): PreparedSparseMatrix = PreparedSparseMatrix(a, this)

    override fun routeOf(operation: SparseMatrixOperation, call: SparseCall): SparseMatrixRoute {
        if (operation == SparseMatrixOperation.Prepare) {
            return route(
                operation,
                RouteKind.Direct,
                emptyList(),
                "a snapshot is copied; no arithmetic kernel runs",
            )
        }
        val scaling = destinationScaling(operation, call)
        if (noWork(operation, call)) {
            return route(
                operation,
                RouteKind.NoWork,
                scaling,
                "the call's own contract stops before the arithmetic, so only the destination scaling runs",
            )
        }
        return when (operation) {
            SparseMatrixOperation.Gemv -> scatteredRoute(operation, call, scaling)

            SparseMatrixOperation.GemvTransposed -> route(
                operation,
                RouteKind.Direct,
                scaling + "${ScalarIndexedSparseKernels.name}/${SparseOperation.DotDense.entryPoint}",
                "a transposed CSC reduction is fixed to the ordered scalar dot, which is its accumulation contract",
            )

            SparseMatrixOperation.GemmDenseRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = false,
                "an update whose scaled coefficient is exactly zero is formed by the traversal rather than by " +
                    "the kernel, which skips a zero multiplier",
            )

            SparseMatrixOperation.SymmRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = false,
                "each stored entry of the selected triangle updates two whole dense columns, and one whose " +
                    "scaled coefficient is exactly zero is formed by the traversal rather than by the kernel",
            )

            SparseMatrixOperation.GemmDense -> panelRoute(
                operation,
                call,
                scaling,
                runs = if (call.transposeSparse) PanelRuns.WholeColumnSource else PanelRuns.WholeColumnDestination,
                covered = if (call.transposeSparse) {
                    "every stored entry of a column reduces into the panel's accumulators"
                } else {
                    "every stored entry of a column scatters across the panel's right-hand sides"
                },
            )

            SparseMatrixOperation.SymmLeft -> panelRoute(
                operation,
                call,
                scaling,
                runs = PanelRuns.Symmetric,
                covered = "each selected run of a column is scattered into the rows it stores and gathered " +
                    "back into its own, which is two panels per column",
            )

            SparseMatrixOperation.TrsmLeft, SparseMatrixOperation.TrmmLeft -> panelRoute(
                operation,
                call,
                scaling,
                runs = if (operation == SparseMatrixOperation.TrsmLeft) {
                    PanelRuns.TriangularSolve
                } else {
                    PanelRuns.TriangularMultiply
                },
                covered = "the strictly triangular part of every column is one panel, and the pivot itself is " +
                    "the traversal's own division or multiplication",
            )

            SparseMatrixOperation.TrsmRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = false,
                "a stored zero coefficient skips its update and the diagonal is divided by in the traversal",
            )

            SparseMatrixOperation.TrmmRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = true,
                "a stored zero coefficient skips its update",
            )

            else -> route(
                operation,
                RouteKind.Direct,
                scaling,
                "the arithmetic is the CSC traversal's own; beyond any destination scaling no kernel is called",
            )
        }
    }

    @Suppress("LongParameterList") // the operation, how it was served, and the facts a report publishes
    private fun route(
        operation: SparseMatrixOperation,
        kind: RouteKind,
        components: List<String>,
        reason: String?,
        group: Int = 0,
        tail: Int = 0,
        resolved: Boolean = true,
    ): SparseMatrixRoute = SparseMatrixRoute(
        operation,
        kind,
        SPARSE_SCHEDULING,
        operation.entryPoint,
        components,
        reason,
        group,
        tail,
        resolved,
    )

    /**
     * Whether the call's own contract stops before any arithmetic.
     *
     * A zero multiplier, an empty operand or an empty destination all end the call after the destination has
     * been scaled, which is the one component such a call still executes.
     */
    private fun noWork(operation: SparseMatrixOperation, call: SparseCall): Boolean {
        val a = call.matrix
        // An empty destination or an empty inner extent is a call with nothing to compute, whatever its
        // operand holds. A destination the operation does not have is not the same as one with no elements,
        // which is why these arrive as null rather than as zero.
        if (call.destinationElements == 0 || call.depth == 0) return true
        val emptyOperand = a.rows == 0 || a.cols == 0
        return when (operation) {
            // An operation returning a fresh structural result still discovers its pattern when the multiplier
            // is zero. The contract says the operand's values are not read, not that its positions are not
            // found, so this is work even though no coefficient is loaded.
            SparseMatrixOperation.GemmSparse, SparseMatrixOperation.AddScaled,
            SparseMatrixOperation.SyrkSparse, SparseMatrixOperation.Transpose,
            -> emptyOperand

            else -> call.alpha == 0.0 || emptyOperand
        }
    }

    /**
     * The destination scaling component, which is a dense Level 1 `scale` when the multiplier is neither zero
     * nor one.
     *
     * A zero multiplier fills rather than scales and a unit one touches nothing, so neither names a kernel.
     * The selected-triangle rank update writes its own triangle element by element and never reaches one
     * either, which is why it is excluded here rather than left to the multiplier.
     */
    private fun destinationScaling(operation: SparseMatrixOperation, call: SparseCall): List<String> {
        val multiplier = when (operation) {
            SparseMatrixOperation.TrsmLeft, SparseMatrixOperation.TrsmRight,
            SparseMatrixOperation.TrmmLeft, SparseMatrixOperation.TrmmRight,
            -> call.alpha

            SparseMatrixOperation.SyrkDense, SparseMatrixOperation.SyrkSparse,
            SparseMatrixOperation.GemmSparse, SparseMatrixOperation.AddScaled,
            SparseMatrixOperation.Transpose, SparseMatrixOperation.Trsv, SparseMatrixOperation.Trmv,
            -> return emptyList()

            else -> call.beta
        }
        val elements = call.destinationElements ?: return emptyList()
        if (multiplier == 0.0 || multiplier == 1.0 || elements == 0) return emptyList()
        return scaleComponent(vectorKernels, elements)
    }

    /**
     * A product that scatters or reduces one CSC column at a time, where the column's own length decides which
     * indexed kernel it reaches.
     *
     * Columns of one matrix need not be the same length, so the implementations they reach need not agree.
     * Naming one of them would publish the other under its label, so a matrix whose columns straddle a
     * crossover is reported as the composition it is.
     */
    private fun scatteredRoute(
        operation: SparseMatrixOperation,
        call: SparseCall,
        scaling: List<String>,
    ): SparseMatrixRoute {
        val leaves = indexedLeaves(call.matrix, SparseOperation.Axpy)
        val entry = SparseOperation.Axpy.entryPoint
        if (leaves.size == 1) {
            return route(
                operation,
                RouteKind.Direct,
                scaling + "${leaves.single()}/$entry",
                "every stored column reaches the same indexed kernel at its own length",
            )
        }
        return route(
            operation,
            RouteKind.Composed,
            scaling + leaves.map { "$it/$entry" },
            "this matrix's column lengths straddle the indexed crossover, so its columns reach " +
                leaves.joinToString(" and "),
        )
    }

    /**
     * A call whose unit of work is a group of right-hand sides handed to a panel leaf.
     *
     * The staging decision is the one the call itself makes, asked here through the same function, so a route
     * cannot name an adjacent panel for a call that will run a strided one. Both widths a call uses are
     * asked about, the plan's and whatever the last group is left with, because a short last group may reach
     * a different body from the full ones.
     */
    private fun panelRoute(
        operation: SparseMatrixOperation,
        call: SparseCall,
        scaling: List<String>,
        runs: PanelRuns,
        covered: String,
    ): SparseMatrixRoute {
        val a = call.matrix
        // A call that reaches here has work to do, since a call without any returned before this. It writes
        // a dense block, so it has right-hand sides, and how many decides the grouping, the staging and the
        // bodies. Without that fact there is nothing to derive, and answering anyway would report a call
        // with no panel for one that is about to run several. The destination extent does not stand in for
        // it: a caller may give one and not the other, or neither.
        if (call.rightHandSides <= 0) {
            return route(
                operation,
                RouteKind.Composed,
                scaling,
                "these call facts carry no right-hand side count, so which panels this call cuts and which " +
                    "bodies they reach is not derivable from them",
                resolved = false,
            )
        }
        val plan = runs.plan(panelKernels, call)
        val staged = rhsStaged(plan)
        val width = rhsWidth(plan)
        val components = ArrayList(scaling)
        if (staged) for (copy in runs.stagedComponents) components.add("$SPARSE_STAGING/$copy")
        val staging = if (staged) runs.stagedReason else ""
        val tail = if (width > 0 && call.rightHandSides > width) call.rightHandSides % width else 0
        // A group of one right-hand side is the panel's arithmetic written out by the traversal, because the
        // seam costs more than it saves on a single value. A last group of one is bypassed exactly as a
        // whole call of one is, so both widths are asked about separately.
        val single = width == 1
        val masked = runs.masked(call.transposeSparse)
        val leaves = if (single) {
            emptyList()
        } else {
            panelLeaves(a, call, width, runs, staged || runs.nativelyContiguous(call))
        }
        val leaf = runs.leaf(call.transposeSparse)
        for (name in leaves) components.add("$name/$leaf")
        val bodies = leaves.joinToString(" and ")
        val (kind, reason) = when {
            single ->
                RouteKind.Direct to
                    "a single right-hand side is written out by the traversal rather than handed to a panel"

            tail == 1 && leaves.isNotEmpty() ->
                RouteKind.Composed to
                    "the full groups reach $bodies" +
                    (
                        if (masked) {
                            " where every right-hand side of a group is live, and keep the written-out loop " +
                                "where one is not"
                        } else {
                            ""
                        }
                        ) +
                    ", and the last group of one is written out by the traversal instead"

            // An empty list is not the same as a call with nothing to do: this one walks its columns and
            // writes its pivots, and what it never reaches is a panel. Naming a body here, vector or
            // otherwise, would report arithmetic that no selected run exists to perform.
            leaves.isEmpty() ->
                RouteKind.Direct to
                    "no column has a selected run to hand over, so no panel body runs at all"

            masked ->
                RouteKind.Composed to
                    "a group whose right-hand sides are all live reaches the panel, and one with a zero " +
                    "right-hand side keeps the written-out loop that skips it"

            leaves.size > 1 -> RouteKind.Composed to "this call's panels reach $bodies"

            else -> RouteKind.Direct to covered
        }
        return route(operation, kind, components, staging + reason, width, tail)
    }

    /**
     * The distinct panel implementations this call's panels reach, in first-seen order.
     *
     * Both extents of a panel are asked about as the call will actually have them: every group of
     * right-hand sides the call cuts, including a short last one, against the length of every run a column
     * really hands over. A triangular routine hands over the strictly triangular part of a column and a
     * symmetric one hands over the selected part in one coupled pass, so a matrix storing only a diagonal,
     * or only the triangle this call does not select, reaches no panel at all and is reported as reaching
     * none.
     */
    private fun panelLeaves(
        a: SparseMatrix,
        call: SparseCall,
        width: Int,
        runs: PanelRuns,
        contiguous: Boolean,
    ): List<String> {
        val reduction = runs.reduction(call.transposeSparse)
        val leaves = ArrayList<String>(2)
        forEachPanelWidth(call.rightHandSides, width) { group ->
            // A group of one is written out by the traversal, so it reaches no panel body to name.
            if (group == 1) return@forEachPanelWidth
            forEachSelectedRun(a, runs, call.lower, runs.forward(call.lower, call.transposeSparse)) { entries ->
                val leaf = panelKernels.panelLeaf(group, entries, contiguous, reduction)
                if (leaf !in leaves) leaves.add(leaf)
            }
        }
        return leaves
    }

    /**
     * The length of every nonempty run of stored entries this call hands to a panel, in the order it hands
     * them over.
     *
     * The order is the traversal's rather than the matrix's. A triangular routine walks its columns in
     * whichever direction its dependence runs, so a route that always scanned forward would list two bodies
     * in the reverse of the order they ran, and first-seen order is what the component list promises.
     */
    private inline fun forEachSelectedRun(
        a: SparseMatrix,
        runs: PanelRuns,
        lower: Boolean,
        forward: Boolean,
        action: (Int) -> Unit,
    ) {
        forEachTriangleColumn(a.cols, forward) { j ->
            val from = a.colPointers[j]
            val to = a.colPointers[j + 1]
            val length = when (runs.shape) {
                RunShape.WholeColumn -> to - from

                RunShape.SelectedRun -> selectedRunEnd(a.rowIndices, from, to, j, lower) -
                    selectedRunStart(a.rowIndices, from, to, j, lower)

                RunShape.StrictRun -> strictRunEnd(a.rowIndices, from, to, j, lower) -
                    strictRunStart(a.rowIndices, from, to, j, lower)
            }
            if (length > 0) action(length)
        }
    }

    /** The distinct indexed implementations the stored columns of [a] reach for [operation], in first-seen order. */
    private fun indexedLeaves(a: SparseMatrix, operation: SparseOperation): List<String> {
        val leaves = ArrayList<String>(2)
        for (j in 0 until a.cols) {
            val length = a.colPointers[j + 1] - a.colPointers[j]
            val leaf = requireNotNull(indexedKernels.implementationFor(operation, length)) {
                "matrix indexed updates must have a shape-determined implementation"
            }
            if (leaf !in leaves) leaves.add(leaf)
        }
        return leaves
    }

    /**
     * A call whose unit of work is a whole contiguous dense column, which is where the selected dense kernels
     * run inside a sparse operation.
     *
     * Always composed: the traversal decides per stored entry whether its column update happens at all, so the
     * named kernel serves some units and not others. Saying which kernel it is, and that the traversal keeps
     * the rest, is the whole of the honest answer.
     */
    private fun wholeColumnRoute(
        operation: SparseMatrixOperation,
        call: SparseCall,
        scaling: List<String>,
        multiplies: Boolean,
        skip: String,
    ): SparseMatrixRoute {
        // Composed rather than direct, because the traversal decides per stored entry whether the named
        // kernel is called at all. The component is what this call can reach, not what it is certain to run.
        val axpy = vectorKernels.implementationFor(DenseOperation.Axpy, call.updateRun)
        val components = ArrayList(scaling)
        if (axpy == null) {
            return route(
                operation,
                RouteKind.Composed,
                components,
                "the dense kernels decide axpy on their own values, so no leaf can be named beforehand",
            )
        }
        components.add("$axpy/axpy")
        if (multiplies) {
            val scale = vectorKernels.implementationFor(DenseOperation.Scale, call.updateRun)
            if (scale != null) components.add("$scale/scale")
        }
        return route(operation, RouteKind.Composed, components, skip)
    }

    @Suppress("LongParameterList") // the BLAS dgemv signature plus the workspace
    override fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        // The no-read shortcut comes first: a zero multiplier or a zero extent reads neither operand, and
        // scaling the destination is all that is left to do.
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
            applyBeta(vectorKernels, y, 0, y.size, beta)
            return
        }
        // Then the snapshots, before the destination is written. Either operand may be the destination's own
        // buffer, and scaling it first would feed the product values the caller never supplied.
        staged(workspace, x, x === y) { stableX ->
            staged(workspace, a, a.values === y) { stableA ->
                gemvCore(alpha, stableA, stableX, beta, y, transpose)
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dgemv signature, over the snapshots taken for it
    private fun gemvCore(
        alpha: Double,
        stableA: SparseMatrix,
        stableX: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        applyBeta(vectorKernels, y, 0, y.size, beta)
        if (transpose) {
            for (j in 0 until stableA.cols) {
                val start = stableA.colPointers[j]
                // CSC gemv promises one input-order accumulation chain; a raw dot permits backend reassociation.
                val sum = ScalarIndexedSparseKernels.dotDense(
                    stableA.rowIndices,
                    start,
                    stableA.values,
                    start,
                    stableA.colPointers[j + 1] - start,
                    stableX,
                )
                y[j] += alpha * sum
            }
        } else {
            for (j in 0 until stableA.cols) {
                val xj = alpha * stableX[j]
                val start = stableA.colPointers[j]
                indexedKernels.axpy(
                    stableA.rowIndices,
                    start,
                    stableA.values,
                    start,
                    stableA.colPointers[j + 1] - start,
                    xj,
                    y,
                )
            }
        }
    }

    override fun transpose(a: SparseMatrix): SparseMatrix = transposeCsc(a)

    @Suppress("LongParameterList") // the BLAS dsymv signature plus the workspace
    override fun symv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSymvOperands(a, x.size, y.size)
        if (alpha == 0.0) {
            applyBeta(vectorKernels, y, 0, y.size, beta)
            return
        }
        staged(workspace, x, x === y) { stableX ->
            staged(workspace, a, a.values === y) { stableA ->
                applyBeta(vectorKernels, y, 0, y.size, beta)
                for (column in 0 until stableA.cols) {
                    panelKernels.symmetricVectorColumn(
                        alpha,
                        column,
                        stableA.rowIndices,
                        stableA.values,
                        stableA.colPointers[column],
                        stableA.colPointers[column + 1],
                        stableX,
                        y,
                        lower,
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymm signature plus the workspace
    override fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        requireSymmOperands(a, b, c, right)
        // A destination with no elements is validated and then left alone, before any staging or loan. One
        // of its two extents may be zero while the other is enormous, and a traversal over the long one
        // would step through it to write nothing.
        if (c.values.isEmpty()) return
        if (alpha == 0.0) {
            applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
            return
        }
        staged(workspace, a, a.values === c.values) { stableA ->
            staged(workspace, b, b.values === c.values) { stableB ->
                applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
                if (right) {
                    for (column in 0 until stableA.cols) {
                        panelKernels.symmetricRightColumn(
                            alpha,
                            column,
                            stableA.rowIndices,
                            stableA.values,
                            stableA.colPointers[column],
                            stableA.colPointers[column + 1],
                            stableB.values,
                            c.values,
                            stableB.rows,
                            lower,
                        )
                    }
                } else {
                    multiplySymmetricFromTheLeft(panelKernels, alpha, stableA, stableB, c, lower, workspace)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrsv signature plus the workspace
    override fun trsv(
        a: SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        // The substitution overwrites x as it goes, so a triangle sharing that buffer is snapshotted first:
        // every column it has yet to reach must still hold the coefficients the caller supplied.
        staged(workspace, a, a.values === x) { stable ->
            trsvCore(panelKernels, stable, x, lower, transpose, unitDiag)
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmv signature plus the workspace
    override fun trmv(
        a: SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        staged(workspace, a, a.values === x) { stable ->
            trmvCore(panelKernels, stable, x, lower, transpose, unitDiag)
        }
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature, plus the side the sparse operand sits on
    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        right: Boolean,
        workspace: Workspace?,
    ) {
        // Multiplying the dense operand by the sparse one from the right is this product with the operands
        // the other way round, so the same derivation answers both.
        if (right) {
            requireGemmOperands(b, transposeB, a, transposeA, c)
        } else {
            requireGemmOperands(a, transposeA, b, transposeB, c)
        }
        // The destination's extents are the product's once that holds, and the depth is the sparse operand's
        // own inner one on the left and the dense operand's on the right.
        val m = c.rows
        val n = c.cols
        val k = if (right) {
            if (transposeB) b.rows else b.cols
        } else {
            if (transposeA) a.rows else a.cols
        }
        // As in symm: nothing to write is settled after validation and before anything is staged or lent.
        if (c.values.isEmpty()) return
        if (alpha == 0.0) {
            applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
            return
        }
        staged(workspace, a, a.values === c.values) { stableA ->
            staged(workspace, b, b.values === c.values) { stableB ->
                applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
                if (right) {
                    multiplyFromTheRight(
                        panelKernels, alpha, stableA, transposeA, stableB, transposeB, c, m, workspace,
                    )
                } else {
                    multiplyFromTheLeft(
                        panelKernels, alpha, stableA, transposeA, stableB, transposeB, c, m, n, k, workspace,
                    )
                }
            }
        }
    }

    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
    ): SparseMatrix {
        requireProductOperands(a, transposeA, b, transposeB, "gemm")
        val aRows = if (transposeA) a.cols else a.rows
        val bCols = if (transposeB) b.rows else b.cols
        // Answered before either operand is oriented, because orienting allocates one pointer per row of the
        // operand and an operand may have more rows than an array can index even when the product it takes
        // part in is empty. Nothing stored on either side reaches no position either.
        if (a.nnz == 0 || b.nnz == 0 || aRows == 0 || bCols == 0) return emptyResult(aRows, bCols, "gemm")
        val left = oriented(a, transposeA, alpha != 0.0)
        val right = oriented(b, transposeB, alpha != 0.0)
        return multiplySparse(left, right, alpha)
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature plus the workspace
    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: SparseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        requireGemmOperands(a, transposeA, b, transposeB, c)
        if (alpha == 0.0) {
            applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
            return
        }
        if (a.nnz == 0 || b.nnz == 0) {
            // No position is reached, so the destination scaling is the whole of the answer and neither
            // operand is oriented. The same reasoning as the fresh sparse product above.
            applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
            return
        }
        staged(workspace, a, a.values === c.values) { stableA ->
            staged(workspace, b, b.values === c.values) { stableB ->
                val left = oriented(stableA, transposeA, true)
                val right = oriented(stableB, transposeB, true)
                applyBeta(vectorKernels, c.values, 0, c.values.size, beta)
                multiplySparseInto(alpha, left, right, c)
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus the workspace
    override fun syrk(
        alpha: Double,
        a: SparseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val n = if (transpose) a.cols else a.rows
        requireSyrkOperands(a, transpose, c)
        if (alpha == 0.0) {
            scaleTriangle(c, n, beta, lower)
            return
        }
        if (a.nnz == 0 || n == 0) {
            // Nothing is stored to reach a position with, so the triangle is scaled and the row adjacency
            // that would have found one is never built. Its scratch is indexed by the source's rows, which a
            // matrix may have more of than an array can hold.
            scaleTriangle(c, n, beta, lower)
            return
        }
        staged(workspace, a, a.values === c.values) { stableA ->
            scaleTriangle(c, n, beta, lower)
            withSymmetricRankScratch(workspace, n, stableA.rows, stableA.nnz) {
                    sums,
                    touchedAt,
                    touched,
                    rowPointers,
                    adjacentColumns,
                    adjacentPositions,
                    rowCursor,
                ->
                symmetricRankInto(
                    alpha, stableA, transpose, c, lower, sums, touchedAt, touched,
                    rowPointers, adjacentColumns, adjacentPositions, rowCursor,
                )
            }
        }
    }

    override fun syrk(a: SparseMatrix, transpose: Boolean, lower: Boolean): SparseMatrix =
        symmetricRankProduct(a, transpose, lower)

    override fun addScaled(alpha: Double, a: SparseMatrix, transposeA: Boolean, b: SparseMatrix): SparseMatrix {
        val rows = if (transposeA) a.cols else a.rows
        val cols = if (transposeA) a.rows else a.cols
        requireSameShape(rows, cols, b, "addScaled")
        // Neither side contributes a position, so the union is empty and no orientation is built for it.
        if (a.nnz == 0 && b.nnz == 0) return emptyResult(rows, cols, "addScaled")
        val left = oriented(a, transposeA, alpha != 0.0)
        val union = left.nnz.toLong() + b.nnz
        requireShape(
            union <= Int.MAX_VALUE,
        ) { "scaled addition needs $union stored entries, more than one array can hold" }
        val pointers = IntArray(left.cols + 1)
        val rowIndices = IntArray(union.toInt())
        val values = DoubleArray(rowIndices.size)
        var count = 0
        for (column in 0 until left.cols) {
            count += SparseAccumulationKernels.mergeScaledColumns(
                alpha,
                left.rowIndices,
                left.values,
                left.colPointers[column],
                left.colPointers[column + 1],
                b.rowIndices,
                b.values,
                b.colPointers[column],
                b.colPointers[column + 1],
                rowIndices,
                values,
                count,
            )
            pointers[column + 1] = count
        }
        return SparseMatrix.wrapTrusted(
            left.rows,
            left.cols,
            pointers,
            rowIndices.copyOf(count),
            values.copyOf(count),
        )
    }

    @Suppress("LongParameterList") // the BLAS dtrsm signature plus the workspace
    override fun trsm(
        a: SparseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        requireTriangularMatrixOperands(a, b, right, "trsm")
        val n = a.rows
        if (alpha == 0.0) {
            b.values.fill(0.0)
            return
        }
        if (n == 0) return
        val rightHandSides = if (right) b.rows else b.cols
        if (rightHandSides == 0) return
        // The triangle is snapshotted before alpha scales the block, not after: a triangle sharing the block's
        // buffer would otherwise be solved against its own scaled coefficients.
        staged(workspace, a, a.values === b.values) { triangle ->
            if (alpha != 1.0) vectorKernels.scale(b.values, 0, alpha, b.values.size)
            if (right) {
                withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                    trsmRightCore(panelKernels, triangle, b, lower, !transpose, diagonal)
                }
            } else {
                withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                    triangularLeftCore(
                        panelKernels, true, triangle, b, lower, transpose, unitDiag, diagonal, workspace,
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature plus the workspace
    override fun trmm(
        a: SparseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        requireTriangularMatrixOperands(a, b, right, "trmm")
        val n = a.rows
        if (alpha == 0.0) {
            b.values.fill(0.0)
            return
        }
        if (n == 0) return
        staged(workspace, a, a.values === b.values) { triangle ->
            if (alpha != 1.0) vectorKernels.scale(b.values, 0, alpha, b.values.size)
            // The diagonal is read once for every right-hand side rather than once per column, as trsm does.
            withExplicitDiagonal(triangle, n, unitDiag, workspace) { diagonal ->
                if (right) {
                    trmmRightCore(panelKernels, triangle, b, lower, transpose, unitDiag, diagonal)
                } else {
                    triangularLeftCore(
                        panelKernels, false, triangle, b, lower, transpose, unitDiag, diagonal, workspace,
                    )
                }
            }
        }
    }

    /**
     * A CSC result of this shape with nothing stored, for a call whose operands reach no position. [what]
     * names the operation in the one error this raises, a column count no array of pointers can hold.
     */
    private fun emptyResult(rows: Int, cols: Int, what: String): SparseMatrix = SparseMatrix.wrapTrusted(
        rows,
        cols,
        IntArray(pointerLength(cols, what)),
        IntArray(0),
        DoubleArray(0),
    )

    /** `beta · C` over exactly the selected triangle, honoring the zero-beta overwrite convention. */
    private fun scaleTriangle(c: DenseMatrix, n: Int, beta: Double, lower: Boolean) {
        if (beta == 1.0) return
        for (j in 0 until n) {
            val first = if (lower) j else 0
            val last = if (lower) n else j + 1
            for (i in first until last) {
                val at = i + j * n
                c.values[at] = if (beta == 0.0) 0.0 else beta * c.values[at]
            }
        }
    }

    /**
     * [a] in the requested orientation, materializing a transpose only where one is asked for.
     *
     * A zero alpha discovers the same structure without reading coefficients, so the transposed pattern is
     * built over zeroed values rather than over the caller's: the contract says alpha of zero reads no
     * operand value, and a transpose that copied them would break it before the product ever ran. The
     * structure is shared rather than copied, because the transpose reads it and writes a fresh result.
     */
    private fun oriented(a: SparseMatrix, transpose: Boolean, readValues: Boolean): SparseMatrix {
        if (!transpose) return a
        if (readValues) return transposeCsc(a)
        return transposeCsc(SparseMatrix.wrapTrusted(a.rows, a.cols, a.colPointers, a.rowIndices, DoubleArray(a.nnz)))
    }
}
