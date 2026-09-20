package com.eignex.koblas.sparse

import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.PanelWork

/**
 * Portable sparse column and right-hand-side panel arithmetic.
 *
 * A panel here is a group of dense right-hand sides visited together, so one walk of a sparse column's
 * indices and values serves several of them. How many is [rightHandSideGroup], which the local backend
 * recommends through the same seam the dense panels use; a caller never has to know that number either.
 *
 * Where the arithmetic over such a group goes is the point of this file. The traversal, the indices and the
 * structural rules are written here and are portable; the arithmetic over one group of right-hand sides is
 * handed to [DensePanelKernels.indexedColumnUpdate] and [DensePanelKernels.indexedRankUpdate], which are the
 * seam's own leaves and the only place a backend's choice of body enters a sparse matrix product. The
 * right-hand sides are the axis with nothing to carry between their entries, so they are what a backend may
 * vectorise; the indices are read as indices on every backend.
 *
 * A dense block reaches these panels as three numbers: an entry is at `offset + rhs · rhsStride +
 * index · indexStride`. That covers both layouts a sparse product meets. A block in the caller's
 * column-major storage has its right-hand sides a leading dimension apart and its indexed axis adjacent; one
 * staged into right-hand-side-major order has them the other way round. Only the second reaches a vector
 * body, which is why staging is a decision the scheduling makes rather than a layout assumed here.
 *
 * Contiguous whole-column work is where a selected Level 1 kernel is called instead, which is the other way
 * an engine's choice of dense kernels reaches a sparse matrix operation.
 */
internal class SparsePanelKernels(
    private val denseVectors: DenseVectorKernels,
    private val densePanels: DensePanelKernels,
) {

    /**
     * Dense right-hand sides to visit per walk of a sparse column, for a destination of [columns] of them.
     *
     * The sparse traversal is this file's, and the indices it reads are not something the dense panel
     * contract knows about. What it takes from that contract is the grouping, so that this width follows the
     * machine the same way the dense ones do instead of being a constant here. [contiguous] is part of the
     * question because a backend whose body over adjacent right-hand sides is a different body may want a
     * different number of them in it; what the scheduling adds on top is a ceiling of its own, for the
     * staging buffer a copy has to fit in.
     */
    fun rightHandSideGroup(rows: Int, columns: Int, contiguous: Boolean = false, reduction: Boolean = false): Int =
        densePanels.executionGroup(panelWork(reduction), rows, columns, contiguous)

    /**
     * Whether this backend does better over a panel of [width] adjacent right-hand sides than over a strided
     * one, which is what decides whether staging a dense block is worth considering at all.
     *
     * The backend's own answer rather than a comparison between two of its labels. A portable backend reads
     * one entry at a time either way and says no, so it never pays for a copy; a vector one says yes where
     * it has a whole lane block of them, and [panelLeaf] then names the body the copy was made for.
     */
    fun prefersAdjacentSides(width: Int, entries: Int, reduction: Boolean = false): Boolean =
        densePanels.prefersContiguous(panelWork(reduction), width, entries)

    /** The Level 1 implementation a contiguous column update of [length] elements reaches, for attribution. */
    fun denseLeaf(operation: DenseOperation, length: Int): String? = denseVectors.implementationFor(operation, length)

    /**
     * The panel body a group of [width] right-hand sides reaches, adjacent or a leading dimension apart.
     *
     * Asked by the route so that a report names the body that ran, with [entries] the stored entries of the
     * run that panel is actually handed rather than the whole column it was cut from.
     */
    fun panelLeaf(width: Int, entries: Int, contiguous: Boolean, reduction: Boolean = false): String =
        densePanels.implementationFor(panelWork(reduction), width, entries, contiguous)

    /**
     * Which of the two sparse panel shapes a caller is asking about.
     *
     * The distinction is the backend's to act on rather than this file's: a group of right-hand sides
     * behaves differently where each of them carries an accumulator from where each of them carries a
     * window of a destination, and only the backend knows what its bodies do with either.
     */
    private fun panelWork(reduction: Boolean): PanelWork =
        if (reduction) PanelWork.SparseRightHandSideReduction else PanelWork.SparseRightHandSides

    @Suppress("LongParameterList") // one CSC column slice, its triangle flag, and the two vectors
    fun symmetricVectorColumn(
        alpha: Double,
        column: Int,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        x: DoubleArray,
        y: DoubleArray,
        lower: Boolean,
    ) {
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            if (if (lower) row >= column else row <= column) {
                val value = values[position]
                y[row] += alpha * (value * x[column])
                if (row != column) y[column] += alpha * (value * x[row])
            }
        }
    }

    /**
     * One CSC column of a symmetric product against a panel of right-hand sides.
     *
     * The stored triangle serves two halves of the product. Every selected entry scatters its column of the
     * dense block into the rows it stores, and every selected entry off the diagonal also gathers those rows
     * back into the column's own row, which is the half the triangle does not store. Both halves run over
     * the same contiguous run of the column's stored entries, because the rows ascend and the selected ones
     * are a prefix or a suffix of them with the diagonal at its inner end, so the two are one call of one
     * panel leaf and the diagonal is the position that leaf leaves out of the second half.
     *
     * One multiplier serves both halves, `alpha · value`, and the mirrored half accumulates into the
     * destination's pivot row as the walk reaches it rather than into a scratch that is added afterwards.
     * Neither half reads a gathered copy, so [width] right-hand sides of [b] serve both and the destination
     * is where every product lands.
     */
    @Suppress("LongParameterList") // one CSC column slice against a dense panel, with both layouts
    fun symmetricLeftColumn(
        alpha: Double,
        column: Int,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        c: DoubleArray,
        offset: Int,
        rhsStride: Int,
        indexStride: Int,
        width: Int,
        lower: Boolean,
    ) {
        val start = selectedRunStart(rowIndices, fromIndex, toIndex, column, lower)
        val end = selectedRunEnd(rowIndices, fromIndex, toIndex, column, lower)
        if (start >= end) return
        val pivot = offset + column * indexStride
        // The diagonal is the one stored entry that scatters and is not gathered back, and an ascending run
        // puts it at the inner end of the selected part if it is stored at all.
        val diagonal = when {
            lower && rowIndices[start] == column -> 0
            !lower && rowIndices[end - 1] == column -> end - start - 1
            else -> -1
        }
        // Both halves read the source through the same layout, so nothing is gathered into a scratch first,
        // and the mirrored half lands in the destination's own pivot row as the walk reaches it: the row is
        // already there, and a column that mirrors nothing leaves it exactly as it was.
        densePanels.indexedCoupledUpdate(
            alpha, c, b, offset, rhsStride, indexStride, rowIndices, values, start, end - start, width,
            pivot, c, pivot, diagonal,
        )
    }

    /**
     * One CSC column of a symmetric product with the dense block on the left, where both halves of the
     * product update whole contiguous dense columns and therefore reach the selected Level 1 `axpy`.
     */
    @Suppress("LongParameterList") // one CSC column slice against a dense block from the right
    fun symmetricRightColumn(
        alpha: Double,
        column: Int,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        c: DoubleArray,
        rows: Int,
        lower: Boolean,
    ) {
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            if (if (lower) row >= column else row <= column) {
                val coefficient = alpha * values[position]
                evaluatedAxpy(c, row * rows, coefficient, b, column * rows, rows)
                if (row != column) evaluatedAxpy(c, column * rows, coefficient, b, row * rows, rows)
            }
        }
    }

    /**
     * One output row of a transposed sparse product, reduced against a panel of dense right-hand sides.
     *
     * The reduction lands in [work], which is adjacent whatever the dense block's layout is, so this half of
     * the product vectorises as soon as the block's own right-hand sides are adjacent. [alpha] scales the
     * finished sum, as it did when this was written out here.
     */
    @Suppress("LongParameterList") // the column slice, the dense window, and the output panel
    fun gatherProductPanel(
        alpha: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        offset: Int,
        rhsStride: Int,
        indexStride: Int,
        width: Int,
        c: DoubleArray,
        cOffset: Int,
        cRhsStride: Int,
        work: DoubleArray,
    ) {
        work.fill(0.0, 0, width)
        densePanels.indexedColumnUpdate(
            1.0, b, offset, rhsStride, indexStride, rowIndices, values, fromIndex, toIndex - fromIndex,
            width, work, 0,
        )
        for (rhs in 0 until width) c[cOffset + rhs * cRhsStride] += alpha * work[rhs]
    }

    /**
     * One inner index of an untransposed sparse product, scattered across a panel of right-hand sides.
     *
     * The coefficients are gathered into [work] first, so the scattered half sees an adjacent source
     * whatever the dense operand's layout is; what decides whether this vectorises is the destination's own,
     * which is why a staged destination is what this direction of the product asks for.
     */
    @Suppress("LongParameterList") // the column slice, the dense window, and the output panel
    fun scatterProductPanel(
        alpha: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        bOffset: Int,
        bRhsStride: Int,
        width: Int,
        c: DoubleArray,
        offset: Int,
        rhsStride: Int,
        indexStride: Int,
        work: DoubleArray,
    ) {
        for (rhs in 0 until width) work[rhs] = alpha * b[bOffset + rhs * bRhsStride]
        densePanels.indexedRankUpdate(
            1.0, c, offset, rhsStride, indexStride, rowIndices, values, fromIndex, toIndex - fromIndex,
            width, work, 0,
        )
    }

    /**
     * The three panels above with one right-hand side, written out.
     *
     * A panel of one is the arithmetic below with the width loop removed, and it is here because the panel
     * machinery is what a narrow call cannot pay for: a group of one spends a call into the seam, a fill and
     * a grouped loop on a single value. A product over one right-hand side measured slower through the
     * panels than the same call before this stage, on columns holding a handful of entries where that
     * overhead is the whole cost; the comparison is in the stage evidence.
     *
     * Every product the panel forms is formed here, over the same entries in the same order, and a reduction
     * is summed as the traversal reaches it. A panel that groups its entries sums them in its own grouping
     * instead, so the two agree to within the reassociation this library allows a grouped product rather
     * than bit for bit. What also changes is that a route names the traversal rather than a panel body,
     * because that is what runs.
     */
    @Suppress("LongParameterList") // the column slice, the dense window, and the single output
    fun gatherProductColumn(
        alpha: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        offset: Int,
        indexStride: Int,
        c: DoubleArray,
        cOffset: Int,
    ) {
        var sum = 0.0
        for (position in fromIndex until toIndex) {
            sum += values[position] * b[offset + rowIndices[position] * indexStride]
        }
        c[cOffset] += alpha * sum
    }

    /** [scatterProductPanel] with one right-hand side, written out for the reason [gatherProductColumn] gives. */
    @Suppress("LongParameterList") // the column slice, the dense window, and the single output
    fun scatterProductColumn(
        alpha: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        bOffset: Int,
        c: DoubleArray,
        offset: Int,
        indexStride: Int,
    ) {
        val coefficient = alpha * b[bOffset]
        for (position in fromIndex until toIndex) {
            c[offset + rowIndices[position] * indexStride] += values[position] * coefficient
        }
    }

    /** [symmetricLeftColumn] with one right-hand side, written out for the same reason. */
    @Suppress("LongParameterList") // one CSC column slice, its triangle flag, and both dense windows
    fun symmetricLeftColumnSingle(
        alpha: Double,
        column: Int,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        c: DoubleArray,
        offset: Int,
        indexStride: Int,
        lower: Boolean,
    ) {
        val start = selectedRunStart(rowIndices, fromIndex, toIndex, column, lower)
        val end = selectedRunEnd(rowIndices, fromIndex, toIndex, column, lower)
        if (start >= end) return
        val pivot = offset + column * indexStride
        val coefficient = b[pivot]
        // One pass for both halves, as the panel makes over a group: the stored entry is read once and
        // spends its one multiplier on the row it scatters into and on the row it mirrors back into, and
        // the diagonal is the one entry that does only the first.
        for (position in start until end) {
            val row = rowIndices[position]
            val t = alpha * values[position]
            val at = offset + row * indexStride
            c[at] += t * coefficient
            if (row != column) c[pivot] += t * b[at]
        }
    }

    /**
     * One CSC column of a product whose sparse operand sits on the right, which is a whole contiguous dense
     * column update per stored entry and therefore the selected Level 1 `axpy`.
     */
    @Suppress("LongParameterList") // the column slice, its orientation, and both dense windows
    fun rightProductColumn(
        alpha: Double,
        column: Int,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        transposeSparse: Boolean,
        b: DoubleArray,
        leadingDimension: Int,
        c: DoubleArray,
        rows: Int,
    ) {
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            val cOffset = (if (transposeSparse) row else column) * rows
            val bColumn = if (transposeSparse) column else row
            evaluatedAxpy(c, cOffset, alpha * values[position], b, bColumn * leadingDimension, rows)
        }
    }

    /** The strictly off-diagonal part of one triangular column, added into a single right-hand side. */
    @Suppress("LongParameterList") // the column slice and its triangle flag
    fun triangularAxpy(
        column: Int,
        lower: Boolean,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        alpha: Double,
        destination: DoubleArray,
    ) {
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            if (if (lower) row > column else row < column) destination[row] += alpha * values[position]
        }
    }

    /** The strictly off-diagonal part of one triangular column, reduced against a single right-hand side. */
    @Suppress("LongParameterList") // the column slice, its triangle flag, and the running sum
    fun triangularReduce(
        column: Int,
        lower: Boolean,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        source: DoubleArray,
        initial: Double,
        subtract: Boolean,
    ): Double {
        var sum = initial
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            if (if (lower) row > column else row < column) {
                val term = values[position] * source[row]
                sum = if (subtract) sum - term else sum + term
            }
        }
        return sum
    }

    /**
     * [triangularPanelScatter] with one right-hand side, written out for the reason [gatherProductColumn]
     * gives.
     *
     * The liveness record the panel keeps is one boolean here, so it is the branch that returns rather than
     * a second half of a scratch array, and the one live right-hand side takes the same arithmetic the panel
     * takes when none of its own are masked.
     */
    @Suppress("LongParameterList") // the column slice, its three triangle flags, and the dense window
    fun triangularScatterSingle(
        solve: Boolean,
        column: Int,
        lower: Boolean,
        unitDiagonal: Boolean,
        diagonal: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        dense: DoubleArray,
        offset: Int,
        indexStride: Int,
    ) {
        val pivot = offset + column * indexStride
        val raw = dense[pivot]
        // A right-hand side that is exactly zero is dead, which leaves the pivot as it stands and spreads
        // nothing; it is the rule the panel records per right-hand side, with one to record.
        if (raw == 0.0) return
        val coefficient = if (solve) raw / diagonal else raw
        dense[pivot] = when {
            solve -> coefficient
            unitDiagonal -> raw
            else -> diagonal * raw
        }
        val start = strictRunStart(rowIndices, fromIndex, toIndex, column, lower)
        val end = strictRunEnd(rowIndices, fromIndex, toIndex, column, lower)
        for (position in start until end) {
            val at = offset + rowIndices[position] * indexStride
            if (solve) {
                dense[at] -= values[position] * coefficient
            } else {
                dense[at] += values[position] * coefficient
            }
        }
    }

    /**
     * [triangularPanelGather] with one right-hand side, written out for the reason [gatherProductColumn]
     * gives.
     *
     * The accumulator is a local rather than an entry of a scratch array, which is the whole of what a group
     * of one has to gain over the panel: the pivot is read once, summed into, and written back once.
     */
    @Suppress("LongParameterList") // the column slice, its three triangle flags, and the dense window
    fun triangularGatherSingle(
        solve: Boolean,
        column: Int,
        lower: Boolean,
        unitDiagonal: Boolean,
        diagonal: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        dense: DoubleArray,
        offset: Int,
        indexStride: Int,
    ) {
        val pivot = offset + column * indexStride
        val raw = dense[pivot]
        var sum = if (solve || unitDiagonal) raw else diagonal * raw
        val start = strictRunStart(rowIndices, fromIndex, toIndex, column, lower)
        val end = strictRunEnd(rowIndices, fromIndex, toIndex, column, lower)
        for (position in start until end) {
            val term = values[position] * dense[offset + rowIndices[position] * indexStride]
            sum = if (solve) sum - term else sum + term
        }
        dense[pivot] = if (solve) sum / diagonal else sum
    }

    /**
     * One triangular column against a panel of right-hand sides, in the direction that finishes the pivot
     * first and then updates the rows below it.
     *
     * A right-hand side that is exactly zero is skipped, which is [SparseBlas.trsv]'s rule carried to several
     * columns at once: it is what keeps a zero right-hand side from forming a product against a stored
     * infinity. Which ones are live is recorded in the second half of [work], so the update loop reads that
     * rather than the values again, and so no width is wider than the record can hold. [work] therefore needs
     * twice [width] entries here, where the other panels need [width].
     *
     * A group whose right-hand sides are all live is handed to the panel leaf instead, because with nothing
     * masked the two agree entry for entry: the skip only ever avoided a product with a zero. A group with
     * one dead right-hand side keeps the written-out loop, whatever the layout, so the rule survives on
     * every backend rather than becoming a property of the one that does not vectorise.
     */
    @Suppress("LongParameterList") // the column slice, its three triangle flags, and the panel window
    fun triangularPanelScatter(
        solve: Boolean,
        column: Int,
        lower: Boolean,
        unitDiagonal: Boolean,
        diagonal: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        dense: DoubleArray,
        offset: Int,
        rhsStride: Int,
        indexStride: Int,
        width: Int,
        work: DoubleArray,
    ) {
        var live = 0
        val pivot = offset + column * indexStride
        for (rhs in 0 until width) {
            val at = pivot + rhs * rhsStride
            val raw = dense[at]
            work[rhs] = when {
                !solve -> raw
                raw == 0.0 -> 0.0
                else -> raw / diagonal
            }
            work[width + rhs] = if (raw == 0.0) 0.0 else 1.0
            if (raw != 0.0) {
                live++
                dense[at] = when {
                    solve -> work[rhs]
                    unitDiagonal -> raw
                    else -> diagonal * raw
                }
            }
        }
        val start = strictRunStart(rowIndices, fromIndex, toIndex, column, lower)
        val end = strictRunEnd(rowIndices, fromIndex, toIndex, column, lower)
        if (start >= end) return
        if (live == width) {
            densePanels.indexedRankUpdate(
                if (solve) -1.0 else 1.0, dense, offset, rhsStride, indexStride,
                rowIndices, values, start, end - start, width, work, 0,
            )
            return
        }
        for (position in start until end) {
            val row = rowIndices[position]
            val value = values[position]
            val target = offset + row * indexStride
            for (rhs in 0 until width) {
                if (work[width + rhs] != 0.0) {
                    val at = target + rhs * rhsStride
                    if (solve) dense[at] -= value * work[rhs] else dense[at] += value * work[rhs]
                }
            }
        }
    }

    /**
     * One triangular column against a panel, in the direction that gathers the finished rows into the pivot.
     *
     * Nothing is masked here: this direction reads the rows it has already finished rather than spreading a
     * pivot into rows it has not, so every right-hand side contributes and the whole strictly triangular run
     * is one panel call. The accumulation lands in [work], which is adjacent whatever the block's layout is.
     */
    @Suppress("LongParameterList") // the column slice, its three triangle flags, and the panel window
    fun triangularPanelGather(
        solve: Boolean,
        column: Int,
        lower: Boolean,
        unitDiagonal: Boolean,
        diagonal: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        dense: DoubleArray,
        offset: Int,
        rhsStride: Int,
        indexStride: Int,
        width: Int,
        work: DoubleArray,
    ) {
        val pivot = offset + column * indexStride
        for (rhs in 0 until width) {
            val raw = dense[pivot + rhs * rhsStride]
            work[rhs] = if (solve || unitDiagonal) raw else diagonal * raw
        }
        val start = strictRunStart(rowIndices, fromIndex, toIndex, column, lower)
        val end = strictRunEnd(rowIndices, fromIndex, toIndex, column, lower)
        if (start < end) {
            densePanels.indexedColumnUpdate(
                if (solve) -1.0 else 1.0, dense, offset, rhsStride, indexStride,
                rowIndices, values, start, end - start, width, work, 0,
            )
        }
        for (rhs in 0 until width) {
            val at = pivot + rhs * rhsStride
            dense[at] = if (solve) work[rhs] / diagonal else work[rhs]
        }
    }

    /**
     * One triangular column with the triangle on the right of the dense block, which makes every update a
     * whole contiguous dense column and therefore the selected Level 1 `axpy` and `scale`.
     *
     * The diagonal is divided by rather than multiplied by its reciprocal, so a right-hand solve rounds the
     * way [SparseBlas.trsv] does over the same triangle.
     */
    @Suppress("LongParameterList") // the column slice, its four flags, and the dense block
    fun triangularRightColumn(
        solve: Boolean,
        gather: Boolean,
        column: Int,
        lower: Boolean,
        unitDiagonal: Boolean,
        diagonal: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        dense: DoubleArray,
        rows: Int,
    ) {
        val columnOffset = column * rows
        if (gather && !unitDiagonal) applyDiagonal(dense, columnOffset, rows, diagonal, solve)
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            val value = values[position]
            if (value != 0.0 && (if (lower) row > column else row < column)) {
                val alpha = if (solve) -value else value
                val rowOffset = row * rows
                if (solve == gather) {
                    denseVectors.axpy(dense, rowOffset, alpha, dense, columnOffset, rows)
                } else {
                    denseVectors.axpy(dense, columnOffset, alpha, dense, rowOffset, rows)
                }
            }
        }
        if (!gather && !unitDiagonal) applyDiagonal(dense, columnOffset, rows, diagonal, solve)
    }

    private fun applyDiagonal(dense: DoubleArray, offset: Int, rows: Int, diagonal: Double, solve: Boolean) {
        if (solve) {
            for (i in 0 until rows) dense[offset + i] = dense[offset + i] / diagonal
        } else {
            denseVectors.scale(dense, offset, diagonal, rows)
        }
    }

    /**
     * `y += alpha · x` over a contiguous run, forming the product even where [alpha] is zero.
     *
     * A matrix product is defined over every position its traversal reaches, so a coefficient that is zero
     * still multiplies its operand: a zero against an infinity is NaN and belongs in the result. Level 1
     * `axpy` is specified the other way round and returns without touching the destination, so the zero case
     * cannot be handed to it.
     */
    @Suppress("LongParameterList") // the two windows and the run length
    private fun evaluatedAxpy(y: DoubleArray, yOffset: Int, alpha: Double, x: DoubleArray, xOffset: Int, length: Int) {
        if (alpha != 0.0) {
            denseVectors.axpy(y, yOffset, alpha, x, xOffset, length)
            return
        }
        for (i in 0 until length) y[yOffset + i] += alpha * x[xOffset + i]
    }
}

/**
 * Where the selected part of a CSC column starts, with the diagonal counted as selected.
 *
 * A column's rows ascend, so what a triangle selects from it is a prefix or a suffix rather than a scattered
 * subset. That is what lets a selected half be handed to a panel leaf as one run, and it is why these four
 * are functions rather than a condition inside a loop: the execution and the route have to agree about
 * which entries a panel is actually given, and they agree by asking the same question here.
 */
internal fun selectedRunStart(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) lowerBound(rowIndices, from, to, column) else from

/** Where the selected part of a column ends, the counterpart of [selectedRunStart]. */
internal fun selectedRunEnd(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) to else lowerBound(rowIndices, from, to, column + 1)

/**
 * Where the strictly triangular part of a column starts, which is the selected part without its diagonal.
 *
 * Without rather than minus: an ascending run has its diagonal at the inner end of the selected part if it
 * is stored at all, so dropping it is a bound one position further in and never a hole in the middle.
 */
internal fun strictRunStart(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) lowerBound(rowIndices, from, to, column + 1) else from

/** Where the strictly triangular part of a column ends, the counterpart of [strictRunStart]. */
internal fun strictRunEnd(rowIndices: IntArray, from: Int, to: Int, column: Int, lower: Boolean): Int =
    if (lower) to else lowerBound(rowIndices, from, to, column)

/**
 * The first position in the ascending run `[from, to)` whose row index is at least [row].
 */
internal fun lowerBound(rowIndices: IntArray, from: Int, to: Int, row: Int): Int {
    // A matrix that stores only the triangle it is asked about answers at one end or the other for every
    // column, so the two endpoints are checked before the run is halved.
    if (from >= to || rowIndices[from] >= row) return from
    if (rowIndices[to - 1] < row) return to
    var low = from + 1
    var high = to - 1
    while (low < high) {
        val middle = (low + high) ushr 1
        if (rowIndices[middle] < row) low = middle + 1 else high = middle
    }
    return low
}
