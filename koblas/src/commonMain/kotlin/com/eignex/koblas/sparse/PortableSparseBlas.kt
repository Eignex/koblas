@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.PreparedSparseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.requireGemmShape
import com.eignex.koblas.requireGemvShape
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSparseProductShape
import com.eignex.koblas.requireSquare
import com.eignex.koblas.requireTriangularMatrixShape
import com.eignex.koblas.sparse.internal.SparseAccumulationKernels
import com.eignex.koblas.sparse.internal.forEachPanelWidth
import com.eignex.koblas.sparse.internal.multiplyFromTheLeft
import com.eignex.koblas.sparse.internal.multiplyFromTheRight
import com.eignex.koblas.sparse.internal.multiplySparse
import com.eignex.koblas.sparse.internal.multiplySparseInto
import com.eignex.koblas.sparse.internal.multiplySymmetricFromTheLeft
import com.eignex.koblas.sparse.internal.planRightHandSides
import com.eignex.koblas.sparse.internal.pointerLength
import com.eignex.koblas.sparse.internal.rhsStaged
import com.eignex.koblas.sparse.internal.rhsWidth
import com.eignex.koblas.sparse.internal.stableFor
import com.eignex.koblas.sparse.internal.symmetricRankInto
import com.eignex.koblas.sparse.internal.symmetricRankProduct
import com.eignex.koblas.sparse.internal.transposeCsc
import com.eignex.koblas.sparse.internal.triangularLeftCore
import com.eignex.koblas.sparse.internal.trmmRightCore
import com.eignex.koblas.sparse.internal.trmvCore
import com.eignex.koblas.sparse.internal.trsmRightCore
import com.eignex.koblas.sparse.internal.trsvCore
import com.eignex.koblas.sparse.internal.withExplicitDiagonal
import com.eignex.koblas.sparse.internal.withSymmetricRankScratch
import com.eignex.koblas.staged
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
 * @property stagedReason how that copy is described in a route's reason.
 * @property writesOutSingleSide whether a group of one right-hand side is written out by the traversal instead
 *   of handed to a panel. Every family is: one column spread into one right-hand side, or reduced into one
 *   output, is a loop the traversal can run without the seam, and the seam costs more than it saves on a
 *   single value. A triangular routine writes out the pivot and the one liveness flag with it, which the
 *   panel keeps in a scratch array because at a group it has one per right-hand side.
 * @param reduces whether this call's panels carry an accumulator per right-hand side rather than a window
 *   of a destination, which is the shape the backend is asked about and may group differently. A triangular
 *   routine answers with its own transpose flag instead, through `reduction`, since the direction that
 *   gathers finished rows into a pivot reduces and the direction that spreads a pivot does not.
 * @param solve which direction a triangular dependence runs, or null where the operation has none.
 */
private enum class PanelRuns(
    val shape: RunShape,
    val stagedComponents: List<String>,
    val stagedReason: String,
    val writesOutSingleSide: Boolean = true,
    private val reduces: Boolean = false,
    private val solve: Boolean? = null,
) {
    /** A product whose dense operand is the source that is staged, and whose destination is an accumulator. */
    WholeColumnSource(
        shape = RunShape.WholeColumn,
        stagedComponents = listOf("rhs-source"),
        stagedReason = "the dense operand is copied into adjacent right-hand sides, which is what lets a " +
            "panel be a vector body, and nothing is written back through it; ",
        reduces = true,
    ),

    /** A product whose destination is the dense block, which is staged and written back. */
    WholeColumnDestination(
        shape = RunShape.WholeColumn,
        stagedComponents = listOf("rhs-destination"),
        stagedReason = "the destination is copied into adjacent right-hand sides and written back, which " +
            "is what lets a panel be a vector body; ",
    ),

    /** A symmetric product, whose selected run is scattered and mirrored back in one coupled pass. */
    Symmetric(
        shape = RunShape.SelectedRun,
        stagedComponents = listOf("rhs-source", "rhs-destination"),
        stagedReason = "both dense blocks are copied into adjacent right-hand sides and the destination is " +
            "written back, which is what lets a panel be a vector body; ",
    ),

    /** A triangular solve, whose panels are the strictly triangular part of a column. */
    TriangularSolve(
        shape = RunShape.StrictRun,
        stagedComponents = listOf("rhs-destination"),
        stagedReason = "the block is copied into adjacent right-hand sides and written back, which is what " +
            "lets a panel be a vector body; ",
        solve = true,
    ),

    /** A triangular multiply, which is the same panels in the other dependence direction. */
    TriangularMultiply(
        shape = RunShape.StrictRun,
        stagedComponents = listOf("rhs-destination"),
        stagedReason = "the block is copied into adjacent right-hand sides and written back, which is what " +
            "lets a panel be a vector body; ",
        solve = false,
    ),
    ;

    /**
     * Whether the columns are visited from the first, which only a triangular routine answers with anything
     * but yes.
     *
     * The same expression the triangular scheduling takes its direction from, because the route has to list
     * the bodies in the order the traversal reached them and a dependence decides that order.
     */
    fun forward(lower: Boolean, transpose: Boolean): Boolean = solve?.let { lower != (transpose == it) } ?: true

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
                rows = productRows(call),
                copiedPerSide = if (call.transposeSparse) productDepth(call).toLong() else 2L * productRows(call),
                nativelyContiguous = call.transposeSparse && call.transposeDense,
                runs = if (call.transposeSparse) PanelRuns.WholeColumnSource else PanelRuns.WholeColumnDestination,
                leaf = if (call.transposeSparse) GATHER_PANEL else SCATTER_PANEL,
                masked = false,
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
                rows = call.matrix.rows,
                copiedPerSide = 3L * call.matrix.rows,
                nativelyContiguous = false,
                runs = PanelRuns.Symmetric,
                leaf = SYMMETRIC_PANEL,
                masked = false,
                covered = "each selected run of a column is scattered into the rows it stores and gathered " +
                    "back into its own, which is two panels per column",
            )

            SparseMatrixOperation.TrsmLeft, SparseMatrixOperation.TrmmLeft -> panelRoute(
                operation,
                call,
                scaling,
                rows = call.matrix.rows,
                copiedPerSide = 2L * call.matrix.rows,
                nativelyContiguous = false,
                runs = if (operation == SparseMatrixOperation.TrsmLeft) {
                    PanelRuns.TriangularSolve
                } else {
                    PanelRuns.TriangularMultiply
                },
                leaf = if (call.transposeSparse) GATHER_PANEL else SCATTER_PANEL,
                masked = !call.transposeSparse,
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
        val leaf = vectorKernels.implementationFor(DenseOperation.Scale, elements)
            ?: return listOf("${vectorKernels.name}/scale")
        return listOf("$leaf/scale")
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
    @Suppress("LongParameterList") // the call, its two extents, the staging inputs and the two descriptions
    private fun panelRoute(
        operation: SparseMatrixOperation,
        call: SparseCall,
        scaling: List<String>,
        rows: Int,
        copiedPerSide: Long,
        nativelyContiguous: Boolean,
        runs: PanelRuns,
        leaf: String,
        masked: Boolean,
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
        // The same question the call itself asks, with the same panel shape: a route that left the shape
        // out would report the width the other one of the two would have taken.
        val plan = planRightHandSides(
            panelKernels,
            rows,
            call.rightHandSides,
            a.nnz,
            copiedPerSide,
            nativelyContiguous,
            runs.reduction(call.transposeSparse),
        )
        val staged = rhsStaged(plan)
        val width = rhsWidth(plan)
        val components = ArrayList(scaling)
        if (staged) for (copy in runs.stagedComponents) components.add("$SPARSE_STAGING/$copy")
        val staging = if (staged) runs.stagedReason else ""
        val tail = if (width > 0 && call.rightHandSides > width) call.rightHandSides % width else 0
        // A group of one is the panel's arithmetic written out by the traversal, because the seam costs
        // more than it saves on a single value. Which operations do that is the operation's own fact: a
        // product spreads or reduces one column into one right-hand side directly, while a triangular
        // routine keeps its panel at every width, since the pivot it writes and the liveness it records are
        // that panel's work whatever its width. A last group of one is bypassed exactly as a whole call of
        // one is, so both widths are asked about separately.
        val bypassed = if (runs.writesOutSingleSide) 1 else 0
        if (runs.writesOutSingleSide && width == 1) {
            return route(
                operation,
                RouteKind.Direct,
                components,
                staging + "a single right-hand side is written out by the traversal rather than handed to a " +
                    "panel",
                width,
                tail,
            )
        }
        val leaves = panelLeaves(
            a,
            call,
            width,
            runs,
            staged || nativelyContiguous,
            bypassed,
            runs.reduction(call.transposeSparse),
        )
        for (name in leaves) components.add("$name/$leaf")
        if (bypassed != 0 && tail == bypassed && leaves.isNotEmpty()) {
            return route(
                operation,
                RouteKind.Composed,
                components,
                staging + "the full groups reach " + leaves.joinToString(" and ") +
                    (
                        if (masked) {
                            " where every right-hand side of a group is live, and keep the " +
                                "written-out loop where one of them is not"
                        } else {
                            ""
                        }
                        ) +
                    ", and the last group of one is written out by the traversal instead",
                width,
                tail,
            )
        }
        if (leaves.isEmpty()) {
            // Not the same as a call with nothing to do: this one walks its columns and writes its pivots,
            // and what it never reaches is a panel. Naming a body here, vector or otherwise, would be
            // reporting arithmetic that no selected run exists to perform.
            return route(
                operation,
                RouteKind.Direct,
                components,
                staging + "no column has a selected run to hand over, so no panel body runs at all",
                width,
                tail,
            )
        }
        if (masked) {
            return route(
                operation,
                RouteKind.Composed,
                components,
                staging + "a group whose right-hand sides are all live reaches the panel, and one with a " +
                    "zero right-hand side keeps the written-out loop that skips it",
                width,
                tail,
            )
        }
        if (leaves.size > 1) {
            return route(
                operation,
                RouteKind.Composed,
                components,
                staging + "this call's panels reach " + leaves.joinToString(" and "),
                width,
                tail,
            )
        }
        return route(operation, RouteKind.Direct, components, staging + covered, width, tail)
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
    @Suppress("LongParameterList") // the operand, the call, the geometry and what the traversal keeps
    private fun panelLeaves(
        a: SparseMatrix,
        call: SparseCall,
        width: Int,
        runs: PanelRuns,
        contiguous: Boolean,
        bypassed: Int,
        reduction: Boolean,
    ): List<String> {
        val leaves = ArrayList<String>(2)
        forEachPanelWidth(call.rightHandSides, width) { group ->
            if (group == bypassed) return@forEachPanelWidth
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
        var step = 0
        while (step < a.cols) {
            val j = if (forward) step else a.cols - 1 - step
            step++
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

    /** Rows of `op(A)` for a product, which the sparse operand's own transpose flag decides. */
    private fun productRows(call: SparseCall): Int = if (call.transposeSparse) call.matrix.cols else call.matrix.rows

    /** The inner extent of a product, the counterpart of [productRows]. */
    private fun productDepth(call: SparseCall): Int = if (call.transposeSparse) call.matrix.rows else call.matrix.cols

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

    @Suppress("LongParameterList") // the BLAS dgemv signature
    override fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        requireGemvShape(a, transpose, x.size, y.size)
        // The no-read shortcut comes first: a zero multiplier or a zero extent reads neither operand, and
        // scaling the destination is all that is left to do.
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) {
            applyBeta(vectorKernels, y, 0, y.size, beta)
            return
        }
        // Then the snapshots, before the destination is written. Either operand may be the destination's own
        // buffer, and scaling it first would feed the product values the caller never supplied.
        val stableX = if (x === y) x.copyOf() else x
        val stableA = a.stableFor(y)
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

    override fun symv(alpha: Double, a: SparseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireSquare(a, "symv")
        requireShape(x.size == a.rows) { "symv: x length ${x.size} != ${a.rows}" }
        requireShape(y.size == a.rows) { "symv: y length ${y.size} != ${a.rows}" }
        if (alpha == 0.0) {
            applyBeta(vectorKernels, y, 0, y.size, beta)
            return
        }
        val stableA = a.stableFor(y)
        val stableX = if (x === y) x.copyOf() else x
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
        requireSquare(a, "symm")
        requireShape(c.rows == b.rows && c.cols == b.cols) {
            "symm: C is ${c.rows}x${c.cols} but B is ${b.rows}x${b.cols}"
        }
        if (right) {
            requireShape(b.cols == a.rows) { "symm right: B has ${b.cols} cols, expected ${a.rows}" }
        } else {
            requireShape(b.rows == a.rows) { "symm: B has ${b.rows} rows, expected ${a.rows}" }
        }
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

    override fun trsv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trsv")
        requireShape(x.size == a.rows) { "trsv: x length ${x.size} != ${a.rows}" }
        // The substitution overwrites x as it goes, so a triangle sharing that buffer is snapshotted first:
        // every column it has yet to reach must still hold the coefficients the caller supplied.
        trsvCore(panelKernels, a.stableFor(x), x, lower, transpose, unitDiag)
    }

    override fun trmv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireSquare(a, "trmv")
        requireShape(x.size == a.rows) { "trmv: x length ${x.size} != ${a.rows}" }
        trmvCore(panelKernels, a.stableFor(x), x, lower, transpose, unitDiag)
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
            requireGemmShape(b, transposeB, a, transposeA, c)
        } else {
            requireGemmShape(a, transposeA, b, transposeB, c)
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
        requireSparseProductShape(a, transposeA, b, transposeB)
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
        requireGemmShape(a, transposeA, b, transposeB, c)
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
        requireShape(c.rows == n && c.cols == n) { "syrk: C is ${c.rows}x${c.cols}, expected ${n}x$n" }
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
        requireShape(rows == b.rows && cols == b.cols) {
            "addScaled: op(A) is ${rows}x$cols but B is ${b.rows}x${b.cols}"
        }
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
        val n = requireTriangularMatrixShape(a, b, right, "trsm")
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
        val n = requireTriangularMatrixShape(a, b, right, "trmm")
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
