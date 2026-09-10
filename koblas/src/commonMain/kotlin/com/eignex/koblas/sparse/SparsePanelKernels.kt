package com.eignex.koblas.sparse

import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.axpyArithmetic

/** Numerical leaves for one sparse CSC column against dense vector or right-hand-side storage. */
internal interface SparsePanelKernels {
    @Suppress("LongParameterList")
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
    )

    @Suppress("LongParameterList")
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
    )

    @Suppress("LongParameterList")
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
    )

    @Suppress("LongParameterList")
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
    )

    @Suppress("LongParameterList")
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
    )

    @Suppress("LongParameterList")
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
    )

    @Suppress("LongParameterList")
    fun triangularAxpy(
        column: Int,
        lower: Boolean,
        rowIndices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        toIndex: Int,
        alpha: Double,
        destination: DoubleArray,
    )

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
    ): Double

    @Suppress("LongParameterList")
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
    )

    @Suppress("LongParameterList")
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
    )

    @Suppress("LongParameterList")
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
    )
}

/** Portable sparse panel arithmetic, retaining the exact selected dense update leaf for contiguous columns. */
internal class PortableSparsePanelKernels(
    private val denseVectors: DenseVectorKernels,
    private val densePanels: DensePanelKernels,
) : SparsePanelKernels {
    override fun symmetricVectorColumn(
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

    override fun symmetricLeftColumn(
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

    override fun symmetricRightColumn(
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

    override fun gatherProductPanel(
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
                work[rhs] +=
                    value *
                    b[
                        if (transposeB) {
                            outputColumn + inner * leadingDimension
                        } else {
                            inner +
                                outputColumn * leadingDimension
                        },
                    ]
            }
        }
        for (rhs in 0 until width) c[(columnStart + rhs) * outputRows + outputRow] += alpha * work[rhs]
    }

    override fun scatterProductPanel(
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
            val raw = b[
                if (transposeB) {
                    outputColumn + innerIndex * leadingDimension
                } else {
                    innerIndex +
                        outputColumn * leadingDimension
                },
            ]
            work[rhs] = alpha * raw
        }
        for (position in fromIndex until toIndex) {
            val row = rowIndices[position]
            val value = values[position]
            for (rhs in 0 until width) c[(columnStart + rhs) * outputRows + row] += value * work[rhs]
        }
    }

    override fun rightProductColumn(
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
            axpyArithmetic(densePanels, c, cOffset, alpha * values[position], b, bColumn * leadingDimension, rows)
        }
    }

    override fun triangularAxpy(
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

    override fun triangularReduce(
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

    override fun triangularPanelScatter(
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
        var active = 0
        for (rhs in 0 until width) {
            val at = (columnStart + rhs) * leadingDimension + column
            val raw = dense[at]
            work[rhs] = when {
                !solve -> raw
                raw == 0.0 -> 0.0
                else -> raw / diagonal
            }
            if (raw != 0.0) {
                active = active or (1 shl rhs)
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
                    if (active and (1 shl rhs) != 0) {
                        val at = (columnStart + rhs) * leadingDimension + row
                        if (solve) dense[at] -= value * work[rhs] else dense[at] += value * work[rhs]
                    }
                }
            }
        }
    }

    override fun triangularPanelGather(
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

    override fun triangularRightColumn(
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
        if (gather && !unitDiagonal) {
            denseVectors.scale(dense, columnOffset, if (solve) 1.0 / diagonal else diagonal, rows)
        }
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
        if (!gather && !unitDiagonal) {
            denseVectors.scale(dense, columnOffset, if (solve) 1.0 / diagonal else diagonal, rows)
        }
    }
}
