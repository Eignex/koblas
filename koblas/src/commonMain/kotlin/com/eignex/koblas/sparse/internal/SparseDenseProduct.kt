@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, C

package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.dense.borrowOptional
import com.eignex.koblas.dense.forEachPanel
import com.eignex.koblas.sparse.SparsePanelKernels
import com.eignex.koblas.sparse.SparseTuning
import kotlin.math.min

/**
 * How a sparse product with a dense block hands its right-hand sides to a panel leaf.
 *
 * Two things are decided together, because each is only worth having with the other. The width is how many
 * right-hand sides one walk of a sparse column serves; staging is whether the block is copied into
 * right-hand-side-major order first, which is what makes those right-hand sides adjacent and so the only way
 * a backend with a vector body can use one on a column-major operand.
 *
 * The width is the backend's recommendation for the layout the panel will actually have, capped by
 * [SparseTuning.contiguousRhsPanel] where a copy has to be held in a buffer. Those are two different
 * questions and neither is the other's: the backend knows what its body wants, and the scheduling knows how
 * much of a dense block it is willing to keep staged while one sparse column is walked.
 *
 * Staging is chosen where the backend says an adjacent panel is a different proposition from a strided one,
 * and where the arithmetic it accelerates is worth the copy it costs. The portable backend says it is not,
 * so it never stages and pays nothing for the question; a call whose right-hand sides are already adjacent
 * takes the adjacent width with no copy at all.
 *
 * The answer is one number rather than a pair, so that asking costs no allocation on the way into a call:
 * the width, negated where the block is staged. [rhsWidth] and [rhsStaged] read it back.
 *
 * @param kernels the sparse panel seam, which is what asks the backend both of its questions.
 * @param rows rows of the destination, which is part of the grouping question a backend is asked.
 * @param sides right-hand sides the call has.
 * @param entries stored entries the traversal will walk for each of them.
 * @param copiedPerSide dense elements a staged panel copies per right-hand side, counting a destination
 *   twice because it is read in and written back. In `Long`, since an extent and a count that are both
 *   valid can multiply past what an `Int` holds, and a wrapped cost would invert the decision it feeds.
 * @param nativelyContiguous whether the caller's own layout already has its right-hand sides adjacent, in
 *   which case the adjacent width is taken without any copy at all.
 * @param reduction which of the two sparse panel shapes this call runs, since a group of right-hand sides
 *   that each carry an accumulator is a different proposition from a group that each carry a window of a
 *   destination. The backend answers for both; this only says which one is being asked about.
 */
@Suppress("LongParameterList") // the seam, the two extents, the copy's cost and what the direction admits
internal fun planRightHandSides(
    kernels: SparsePanelKernels,
    rows: Int,
    sides: Int,
    entries: Int,
    copiedPerSide: Long,
    nativelyContiguous: Boolean,
    reduction: Boolean = false,
): Int {
    if (sides <= 0) return 0
    val adjacent = min(
        sides,
        min(kernels.rightHandSideGroup(rows, sides, contiguous = true, reduction = reduction), stagedPanelCap()),
    )
    if (nativelyContiguous) return adjacent
    val strided = kernels.rightHandSideGroup(rows, sides, contiguous = false, reduction = reduction)
    if (entries <= 0) return strided
    // A backend that does no better over an adjacent panel gains nothing from the copy.
    if (!kernels.prefersAdjacentSides(adjacent, entries, reduction)) return strided
    // And one that does gains it only where the arithmetic the copy accelerates outweighs the copy itself.
    if (entries.toLong() < SparseTuning.stagedRhsCrossover.toLong() * copiedPerSide) return strided
    return -adjacent
}

/** Right-hand sides the scheduling is willing to hold staged at once, whatever a backend would like. */
private fun stagedPanelCap(): Int = SparseTuning.contiguousRhsPanel

/** The width in a plan from [planRightHandSides]. */
internal fun rhsWidth(plan: Int): Int = if (plan < 0) -plan else plan

/** Whether a plan from [planRightHandSides] stages the dense block. */
internal fun rhsStaged(plan: Int): Boolean = plan < 0

/** The widths a call's right-hand-side panels actually take, which is the plan's and whatever is left over. */
internal inline fun forEachPanelWidth(sides: Int, width: Int, action: (Int) -> Unit) {
    if (sides <= 0 || width <= 0) return
    action(min(width, sides))
    val tail = sides % width
    if (tail != 0 && sides > width) action(tail)
}

/**
 * Copies one panel of right-hand sides into right-hand-side-major order.
 *
 * The destination holds [width] adjacent right-hand sides per index, [stride] apart, which is the layout
 * whose right-hand-side axis a vector body can load. [count] is how far the indexed axis runs.
 */
@Suppress("LongParameterList") // the source window with its two strides, and the staged panel
internal fun stageRhsPanel(
    source: DoubleArray,
    sourceOffset: Int,
    sourceRhsStride: Int,
    sourceIndexStride: Int,
    count: Int,
    width: Int,
    destination: DoubleArray,
    stride: Int,
) {
    for (index in 0 until count) {
        val from = sourceOffset + index * sourceIndexStride
        val to = index * stride
        for (rhs in 0 until width) destination[to + rhs] = source[from + rhs * sourceRhsStride]
    }
}

/**
 * Writes a staged panel back where it came from.
 *
 * Every position is written, including one the arithmetic never reached, and what goes back there is what
 * was read: a destination is staged by copying it rather than by starting from zero, so an untouched entry
 * returns bit for bit and a negative zero stays negative.
 */
@Suppress("LongParameterList") // the destination window with its two strides, and the staged panel
internal fun unstageRhsPanel(
    staged: DoubleArray,
    stride: Int,
    destination: DoubleArray,
    destinationOffset: Int,
    destinationRhsStride: Int,
    destinationIndexStride: Int,
    count: Int,
    width: Int,
) {
    for (index in 0 until count) {
        val to = destinationOffset + index * destinationIndexStride
        val from = index * stride
        for (rhs in 0 until width) destination[to + rhs * destinationRhsStride] = staged[from + rhs]
    }
}

/**
 * `C += alpha · op(A) · op(B)` for a sparse `A` on the left, reusing each walk over a panel of right-hand
 * sides.
 *
 * The transposed operand reduces the rows its column stores into one output row, so its accumulator is
 * adjacent whatever the dense block looks like and only the block's own layout decides whether the panel
 * vectorises. The untransposed one spreads one inner index across the rows it stores, so its destination is
 * the dense block and a column-major one is strided there; that is the direction a staged panel is for.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheLeft(
    kernels: SparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    m: Int,
    n: Int,
    k: Int,
    workspace: Workspace?,
) {
    // Before the plan and before any loan: a call with nothing to write has nothing to stage or walk, and
    // an empty axis may be paired with a long one, where a traversal would step through it for nothing.
    if (m == 0 || n == 0 || k == 0) return
    val ldb = b.rows
    // The dense operand as (inner, right-hand side), which its own transpose flag decides.
    val bRhsStride = if (transposeB) 1 else ldb
    val bIndexStride = if (transposeB) ldb else 1
    val plan = if (transposeA) {
        planRightHandSides(
            kernels,
            m,
            n,
            a.nnz,
            copiedPerSide = k.toLong(),
            nativelyContiguous = transposeB,
            reduction = true,
        )
    } else {
        planRightHandSides(kernels, m, n, a.nnz, copiedPerSide = 2L * m, nativelyContiguous = false)
    }
    val width = rhsWidth(plan)
    val staged = rhsStaged(plan)
    workspace.borrow(width) { work ->
        workspace.borrowOptional(if (staged) width * (if (transposeA) k else m) else 0) { panel ->
            forEachPanel(n, width) { columnStart, actual ->
                if (transposeA) {
                    gatherPanel(
                        kernels, alpha, a, b, c, m, k, columnStart, actual, width, staged,
                        bRhsStride, bIndexStride, panel, work,
                    )
                } else {
                    scatterPanel(
                        kernels, alpha, a, b, c, m, k, columnStart, actual, width, staged,
                        bRhsStride, bIndexStride, panel, work,
                    )
                }
            }
        }
    }
}

/** One panel of right-hand sides of `C += alpha · Aᵀ · op(B)`, staged or in the caller's own layout. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, the panel window, and the staged buffers
private fun gatherPanel(
    kernels: SparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    c: DenseMatrix,
    m: Int,
    k: Int,
    columnStart: Int,
    actual: Int,
    width: Int,
    staged: Boolean,
    bRhsStride: Int,
    bIndexStride: Int,
    panel: DoubleArray,
    work: DoubleArray,
) {
    val source: DoubleArray
    val offset: Int
    val rhsStride: Int
    val indexStride: Int
    if (staged) {
        stageRhsPanel(b.values, columnStart * bRhsStride, bRhsStride, bIndexStride, k, actual, panel, width)
        source = panel
        offset = 0
        rhsStride = 1
        indexStride = width
    } else {
        source = b.values
        offset = columnStart * bRhsStride
        rhsStride = bRhsStride
        indexStride = bIndexStride
    }
    if (actual == 1) {
        for (outputRow in 0 until m) {
            kernels.gatherProductColumn(
                alpha, a.rowIndices, a.values, a.colPointers[outputRow], a.colPointers[outputRow + 1],
                source, offset, indexStride, c.values, outputRow + columnStart * m,
            )
        }
        return
    }
    for (outputRow in 0 until m) {
        kernels.gatherProductPanel(
            alpha, a.rowIndices, a.values, a.colPointers[outputRow], a.colPointers[outputRow + 1],
            source, offset, rhsStride, indexStride, actual,
            c.values, outputRow + columnStart * m, m, work,
        )
    }
}

/** One panel of right-hand sides of `C += alpha · A · op(B)`, with the destination staged or in place. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, the panel window, and the staged buffers
private fun scatterPanel(
    kernels: SparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    c: DenseMatrix,
    m: Int,
    k: Int,
    columnStart: Int,
    actual: Int,
    width: Int,
    staged: Boolean,
    bRhsStride: Int,
    bIndexStride: Int,
    panel: DoubleArray,
    work: DoubleArray,
) {
    val destination = if (staged) panel else c.values
    val offset = if (staged) 0 else columnStart * m
    val rhsStride = if (staged) 1 else m
    val indexStride = if (staged) width else 1
    if (staged) stageRhsPanel(c.values, columnStart * m, m, 1, m, actual, panel, width)
    if (actual == 1) {
        for (inner in 0 until k) {
            kernels.scatterProductColumn(
                alpha, a.rowIndices, a.values, a.colPointers[inner], a.colPointers[inner + 1],
                b.values, columnStart * bRhsStride + inner * bIndexStride,
                destination, offset, indexStride,
            )
        }
    } else {
        for (inner in 0 until k) {
            kernels.scatterProductPanel(
                alpha, a.rowIndices, a.values, a.colPointers[inner], a.colPointers[inner + 1],
                b.values, columnStart * bRhsStride + inner * bIndexStride, bRhsStride, actual,
                destination, offset, rhsStride, indexStride, work,
            )
        }
    }
    if (staged) unstageRhsPanel(panel, width, c.values, columnStart * m, m, 1, m, actual)
}

/**
 * `C += alpha · A · B` for a symmetric sparse `A` on the left, over panels of right-hand sides.
 *
 * Both dense operands are the same shape and the same layout, so one panel geometry serves the source and
 * the destination. Staging copies both, and writes only the destination back; that is three passes over a
 * right-hand side against the two a general product's destination costs, which is why the two callers weigh
 * the copy differently.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, the triangle flag, and the scratch
internal fun multiplySymmetricFromTheLeft(
    kernels: SparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    c: DenseMatrix,
    lower: Boolean,
    workspace: Workspace?,
) {
    val n = a.rows
    val sides = b.cols
    val ld = b.rows
    if (n == 0 || sides == 0) return
    val plan = planRightHandSides(kernels, n, sides, a.nnz, copiedPerSide = 3L * n, nativelyContiguous = false)
    val width = rhsWidth(plan)
    val staged = rhsStaged(plan)
    val panelSize = if (staged) width * n else 0
    workspace.borrowOptional(panelSize) { source ->
        workspace.borrowOptional(panelSize) { destination ->
            forEachPanel(sides, width) { columnStart, actual ->
                val from = if (staged) source else b.values
                val into = if (staged) destination else c.values
                val offset = if (staged) 0 else columnStart * ld
                val rhsStride = if (staged) 1 else ld
                val indexStride = if (staged) width else 1
                if (staged) {
                    stageRhsPanel(b.values, columnStart * ld, ld, 1, n, actual, source, width)
                    stageRhsPanel(c.values, columnStart * ld, ld, 1, n, actual, destination, width)
                }
                for (column in 0 until n) {
                    if (actual == 1) {
                        kernels.symmetricLeftColumnSingle(
                            alpha, column, a.rowIndices, a.values,
                            a.colPointers[column], a.colPointers[column + 1],
                            from, into, offset, indexStride, lower,
                        )
                    } else {
                        kernels.symmetricLeftColumn(
                            alpha, column, a.rowIndices, a.values,
                            a.colPointers[column], a.colPointers[column + 1],
                            from, into, offset, rhsStride, indexStride, actual, lower,
                        )
                    }
                }
                if (staged) unstageRhsPanel(destination, width, c.values, columnStart * ld, ld, 1, n, actual)
            }
        }
    }
}

/**
 * `C += alpha · op(B) · op(A)` over the sparse operand's CSC columns.
 *
 * With the sparse operand on the right every update is a whole column of the dense block, so a transposed
 * dense operand is staged once into column-major order rather than read across its rows once per entry.
 */
@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheRight(
    kernels: SparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    m: Int,
    workspace: Workspace?,
) {
    if (transposeB) {
        workspace.borrow(b.values.size) { packed ->
            for (j in 0 until b.cols) {
                for (i in 0 until b.rows) packed[j + i * b.cols] = b.values[i + j * b.rows]
            }
            multiplyFromTheRightColumns(kernels, alpha, a, transposeA, c.values, packed, m, m)
        }
    } else {
        multiplyFromTheRightColumns(kernels, alpha, a, transposeA, c.values, b.values, m, b.rows)
    }
}

@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, their flags, and both dense windows
internal fun multiplyFromTheRightColumns(
    kernels: SparsePanelKernels,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    c: DoubleArray,
    b: DoubleArray,
    rows: Int,
    leadingDimension: Int,
) {
    for (column in 0 until a.cols) {
        kernels.rightProductColumn(
            alpha, column, a.rowIndices, a.values, a.colPointers[column], a.colPointers[column + 1], transposeA,
            b, leadingDimension, c, rows,
        )
    }
}
