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
 * recommends through the same seam the dense panels use; it is not a vector width, since nothing below is
 * written in lanes, and a caller never has to know either number.
 *
 * Contiguous whole-column work is where a selected Level 1 kernel is actually called, which is the only place
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
     * machine the same way the dense ones do instead of being a constant here.
     */
    fun rightHandSideGroup(rows: Int, columns: Int): Int =
        densePanels.executionGroup(PanelWork.SparseRightHandSides, rows, columns)

    /** The Level 1 implementation a contiguous column update of [length] elements reaches, for attribution. */
    fun denseLeaf(operation: DenseOperation, length: Int): String? = denseVectors.implementationFor(operation, length)

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

    @Suppress("LongParameterList") // one CSC column slice against a dense block
    fun symmetricLeftColumn(
        alpha: Double,
        column: Int,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        c: DoubleArray,
        rows: Int,
        columns: Int,
        lower: Boolean,
    ) {
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            if (if (lower) row >= column else row <= column) {
                val value = values[position]
                for (rhs in 0 until columns) {
                    val offset = rhs * rows
                    c[offset + row] += alpha * (value * b[offset + column])
                    if (row != column) c[offset + column] += alpha * (value * b[offset + row])
                }
            }
        }
    }

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
                val value = values[position]
                for (rhs in 0 until rows) {
                    c[rhs + row * rows] += alpha * (b[rhs + column * rows] * value)
                    if (row != column) c[rhs + column * rows] += alpha * (b[rhs + row * rows] * value)
                }
            }
        }
    }

    /** One output row of a transposed sparse product, reduced against a panel of dense right-hand sides. */
    @Suppress("LongParameterList") // the column slice, the dense window, and the output panel
    fun gatherProductPanel(
        alpha: Double,
        outputRow: Int,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        leadingDimension: Int,
        transposeB: Boolean,
        columnStart: Int,
        width: Int,
        c: DoubleArray,
        outputRows: Int,
        work: DoubleArray,
    ) {
        work.fill(0.0, 0, width)
        for (position in fromIndex until toIndex) {
            val inner = rowIndices[position]
            val value = values[position]
            for (rhs in 0 until width) {
                val outputColumn = columnStart + rhs
                val at = if (transposeB) {
                    outputColumn + inner * leadingDimension
                } else {
                    inner +
                        outputColumn * leadingDimension
                }
                work[rhs] += value * b[at]
            }
        }
        for (rhs in 0 until width) c[(columnStart + rhs) * outputRows + outputRow] += alpha * work[rhs]
    }

    /** One inner index of an untransposed sparse product, scattered across a panel of right-hand sides. */
    @Suppress("LongParameterList") // the column slice, the dense window, and the output panel
    fun scatterProductPanel(
        alpha: Double,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        b: DoubleArray,
        leadingDimension: Int,
        transposeB: Boolean,
        innerIndex: Int,
        columnStart: Int,
        width: Int,
        c: DoubleArray,
        outputRows: Int,
        work: DoubleArray,
    ) {
        for (rhs in 0 until width) {
            val outputColumn = columnStart + rhs
            val at = if (transposeB) {
                outputColumn + innerIndex * leadingDimension
            } else {
                innerIndex + outputColumn * leadingDimension
            }
            work[rhs] = alpha * b[at]
        }
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            val value = values[position]
            for (rhs in 0 until width) c[(columnStart + rhs) * outputRows + row] += value * work[rhs]
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
     * One triangular column against a panel of right-hand sides, in the direction that finishes the pivot
     * first and then updates the rows below it.
     *
     * A right-hand side that is exactly zero is skipped, which is [SparseBlas.trsv]'s rule carried to several
     * columns at once: it is what keeps a zero right-hand side from forming a product against a stored
     * infinity. Which ones are live is recorded in the second half of [work], so the update loop reads that
     * rather than the values again, and so no width is wider than the record can hold. [work] therefore needs
     * twice [width] entries here, where the other panels need [width].
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
        leadingDimension: Int,
        columnStart: Int,
        width: Int,
        work: DoubleArray,
    ) {
        for (rhs in 0 until width) {
            val at = (columnStart + rhs) * leadingDimension + column
            val raw = dense[at]
            work[rhs] = when {
                !solve -> raw
                raw == 0.0 -> 0.0
                else -> raw / diagonal
            }
            work[width + rhs] = if (raw == 0.0) 0.0 else 1.0
            if (raw != 0.0) {
                dense[at] = when {
                    solve -> work[rhs]
                    unitDiagonal -> raw
                    else -> diagonal * raw
                }
            }
        }
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            if (if (lower) row > column else row < column) {
                val value = values[position]
                for (rhs in 0 until width) {
                    if (work[width + rhs] != 0.0) {
                        val at = (columnStart + rhs) * leadingDimension + row
                        if (solve) dense[at] -= value * work[rhs] else dense[at] += value * work[rhs]
                    }
                }
            }
        }
    }

    /** One triangular column against a panel, in the direction that gathers the finished rows into the pivot. */
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
        leadingDimension: Int,
        columnStart: Int,
        width: Int,
        work: DoubleArray,
    ) {
        for (rhs in 0 until width) {
            val raw = dense[(columnStart + rhs) * leadingDimension + column]
            work[rhs] = if (solve || unitDiagonal) raw else diagonal * raw
        }
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            if (if (lower) row > column else row < column) {
                val value = values[position]
                for (rhs in 0 until width) {
                    val term = value * dense[(columnStart + rhs) * leadingDimension + row]
                    if (solve) work[rhs] -= term else work[rhs] += term
                }
            }
        }
        for (rhs in 0 until width) {
            val at = (columnStart + rhs) * leadingDimension + column
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
