@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C, x, y

package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.scaleComponent
import com.eignex.koblas.sparse.internal.forEachPanelWidth
import com.eignex.koblas.sparse.internal.forEachTriangleColumn
import com.eignex.koblas.sparse.internal.productRhsPlan
import com.eignex.koblas.sparse.internal.rhsStaged
import com.eignex.koblas.sparse.internal.rhsWidth
import com.eignex.koblas.sparse.internal.symmetricRhsPlan
import com.eignex.koblas.sparse.internal.triangularForward
import com.eignex.koblas.sparse.internal.triangularRhsPlan
import com.eignex.koblas.vendor.RouteKind

internal const val SPARSE_SCHEDULING: String = "portable-csc"

internal const val SPARSE_STAGING: String = "portable-stage"

private const val GATHER_PANEL: String = "sparse-rhs-gather"

private const val SCATTER_PANEL: String = "sparse-rhs-scatter"

private const val SYMMETRIC_PANEL: String = "sparse-rhs-mirror"

private enum class PanelRuns(
    val shape: RunShape,
    val stagedComponents: List<String>,
    private val reduces: Boolean = false,
    private val solve: Boolean? = null,
) {
    WholeColumnSource(RunShape.WholeColumn, listOf("rhs-source"), reduces = true),

    WholeColumnDestination(RunShape.WholeColumn, listOf("rhs-destination")),

    Symmetric(RunShape.SelectedRun, listOf("rhs-source", "rhs-destination")),

    TriangularSolve(RunShape.StrictRun, listOf("rhs-destination"), solve = true),

    TriangularMultiply(RunShape.StrictRun, listOf("rhs-destination"), solve = false),
    ;

    fun leaf(transpose: Boolean): String = when {
        this == Symmetric -> SYMMETRIC_PANEL
        reduction(transpose) -> GATHER_PANEL
        else -> SCATTER_PANEL
    }

    fun masked(transpose: Boolean): Boolean = solve != null && !transpose

    fun forward(lower: Boolean, transpose: Boolean): Boolean =
        solve?.let { triangularForward(lower, transpose, it) } ?: true

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

    fun nativelyContiguous(call: SparseCall): Boolean =
        shape == RunShape.WholeColumn && call.transposeSparse && call.transposeDense

    fun reduction(transpose: Boolean): Boolean = if (solve == null) reduces else transpose
}

private enum class RunShape {
    WholeColumn,

    SelectedRun,

    StrictRun,
}

internal class SparseRouteReporter(
    private val vectorKernels: DenseVectorKernels,
    private val indexedKernels: IndexedSparseKernels,
    private val panelKernels: SparsePanelKernels,
) {
    fun routeOf(operation: SparseMatrixOperation, call: SparseCall): SparseMatrixRoute {
        if (operation == SparseMatrixOperation.Prepare) {
            return route(
                operation,
                RouteKind.Direct,
                emptyList(),
            )
        }
        val scaling = destinationScaling(operation, call)
        if (noWork(operation, call)) {
            return route(
                operation,
                RouteKind.NoWork,
                scaling,
            )
        }
        return when (operation) {
            SparseMatrixOperation.Gemv -> scatteredRoute(operation, call, scaling)

            SparseMatrixOperation.GemvTransposed -> scatteredRoute(operation, call, scaling, SparseOperation.DotDense)

            SparseMatrixOperation.GemmDenseRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = false,
            )

            SparseMatrixOperation.SymmRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = false,
            )

            SparseMatrixOperation.GemmDense -> panelRoute(
                operation,
                call,
                scaling,
                runs = if (call.transposeSparse) PanelRuns.WholeColumnSource else PanelRuns.WholeColumnDestination,
            )

            SparseMatrixOperation.SymmLeft -> panelRoute(
                operation,
                call,
                scaling,
                runs = PanelRuns.Symmetric,
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
            )

            SparseMatrixOperation.TrsmRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = false,
            )

            SparseMatrixOperation.TrmmRight -> wholeColumnRoute(
                operation,
                call,
                scaling,
                multiplies = true,
            )

            else -> route(
                operation,
                RouteKind.Direct,
                scaling,
            )
        }
    }

    @Suppress("LongParameterList") // the operation, how it was served, and the facts a report publishes
    private fun route(
        operation: SparseMatrixOperation,
        kind: RouteKind,
        components: List<String>,
        group: Int = 0,
        tail: Int = 0,
        resolved: Boolean = true,
    ): SparseMatrixRoute = SparseMatrixRoute(
        operation,
        kind,
        SPARSE_SCHEDULING,
        operation.entryPoint,
        components,
        group,
        tail,
        resolved,
    )

    private fun noWork(operation: SparseMatrixOperation, call: SparseCall): Boolean {
        val a = call.matrix
        // Null means no destination extent was supplied; zero means the destination is empty.
        if (call.destinationElements == 0 || call.depth == 0) return true
        val emptyOperand = a.rows == 0 || a.cols == 0
        return when (operation) {
            // A transposed matvec skips empty columns before reaching an indexed dot kernel.
            SparseMatrixOperation.GemvTransposed -> call.alpha == 0.0 || emptyOperand || a.nnz == 0

            // A fresh structural result discovers its pattern even at zero alpha, without reading coefficients.
            SparseMatrixOperation.GemmSparse, SparseMatrixOperation.AddScaled,
            SparseMatrixOperation.SyrkSparse, SparseMatrixOperation.Transpose,
            -> emptyOperand

            else -> call.alpha == 0.0 || emptyOperand
        }
    }

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

    private fun scatteredRoute(
        operation: SparseMatrixOperation,
        call: SparseCall,
        scaling: List<String>,
        leafOperation: SparseOperation = SparseOperation.Axpy,
    ): SparseMatrixRoute {
        val leaves = indexedLeaves(call.matrix, leafOperation)
        val entry = leafOperation.entryPoint
        if (leaves.size == 1) {
            return route(
                operation,
                RouteKind.Direct,
                scaling + "${leaves.single()}/$entry",
            )
        }
        return route(
            operation,
            RouteKind.Composed,
            scaling + leaves.map { "$it/$entry" },
        )
    }

    private fun panelRoute(
        operation: SparseMatrixOperation,
        call: SparseCall,
        scaling: List<String>,
        runs: PanelRuns,
    ): SparseMatrixRoute {
        val a = call.matrix
        // Panel routing requires the right-hand-side count; destination extent cannot substitute for it.
        if (call.rightHandSides <= 0) {
            return route(
                operation,
                RouteKind.Composed,
                scaling,
                resolved = false,
            )
        }
        val plan = runs.plan(panelKernels, call)
        val staged = rhsStaged(plan)
        val width = rhsWidth(plan)
        val components = ArrayList(scaling)
        if (staged) for (copy in runs.stagedComponents) components.add("$SPARSE_STAGING/$copy")
        val tail = if (width > 0 && call.rightHandSides > width) call.rightHandSides % width else 0
        // Single-right-hand-side groups bypass the panel seam to avoid its overhead, including a final tail.
        val single = width == 1
        val masked = runs.masked(call.transposeSparse)
        val leaves = if (single) {
            emptyList()
        } else {
            panelLeaves(a, call, width, runs, staged || runs.nativelyContiguous(call))
        }
        val leaf = runs.leaf(call.transposeSparse)
        for (name in leaves) components.add("$name/$leaf")
        val kind = if (leaves.isNotEmpty() && (tail == 1 || masked || leaves.size > 1)) {
            RouteKind.Composed
        } else {
            RouteKind.Direct
        }
        return route(operation, kind, components, width, tail)
    }

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

    private fun indexedLeaves(a: SparseMatrix, operation: SparseOperation): List<String> {
        val leaves = ArrayList<String>(2)
        for (j in 0 until a.cols) {
            val length = a.colPointers[j + 1] - a.colPointers[j]
            if (length == 0 && operation == SparseOperation.DotDense) continue
            val leaf = requireNotNull(indexedKernels.implementationFor(operation, length)) {
                "matrix indexed updates must have a shape-determined implementation"
            }
            if (leaf !in leaves) leaves.add(leaf)
        }
        return leaves
    }

    private fun wholeColumnRoute(
        operation: SparseMatrixOperation,
        call: SparseCall,
        scaling: List<String>,
        multiplies: Boolean,
    ): SparseMatrixRoute {
        // The traversal decides per stored entry whether this component runs, so the route is composed.
        val axpy = vectorKernels.implementationFor(DenseOperation.Axpy, call.updateRun)
        val components = ArrayList(scaling)
        if (axpy == null) {
            return route(
                operation,
                RouteKind.Composed,
                components,
            )
        }
        components.add("$axpy/axpy")
        if (multiplies) {
            val scale = vectorKernels.implementationFor(DenseOperation.Scale, call.updateRun)
            if (scale != null) components.add("$scale/scale")
        }
        return route(operation, RouteKind.Composed, components)
    }
}
