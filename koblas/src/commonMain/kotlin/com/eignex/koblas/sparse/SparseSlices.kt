package com.eignex.koblas.sparse

import com.eignex.koblas.requireIndex
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.internal.SparseAccumulationKernels

/**
 * Stateless, allocation-free sparse arithmetic over caller-owned array slices.
 *
 * The caller owns every buffer and retains it between calls. Scatter input indices and touched indices used by
 * gather or clear operations must be unique within their supplied slices; reductions explicitly permit repeated
 * indices. Required uniqueness, and agreement between an active mark and the touched slice, are caller
 * preconditions: checking either without temporary storage would require quadratic work or a full-dimension scan.
 * Bounds, windows, capacities, overlap, and scatter's incoming indices are validated before any destination is
 * mutated. The existing touched slice is trusted and is never scanned by scatter. Roles without two explicit
 * comparable slices must use distinct arrays; explicit
 * `IntArray` slices may share an array only when their reserved windows do not overlap.
 *
 * Unlike [com.eignex.koblas.Workspace], this object owns and borrows no storage. These helpers manipulate
 * arithmetic values and active support only. Pivot selection, merit calculations,
 * permutations, dropping policy, and factorization state remain caller responsibilities.
 */
public object SparseSlices {
    /** Bit reported by [scatterAxpyChecked] when a product or updated accumulator value is not finite. */
    public const val ARITHMETIC_NONFINITE: Int = 1

    /** Bit reported by [scatterAxpyChecked] when nonzero operands produce a zero product. */
    public const val ARITHMETIC_NONZERO_PRODUCT_UNDERFLOW: Int = 2

    /**
     * Accumulates `alpha * values(k)` at `indices(k)` and returns the new total touched count.
     *
     * A first touch initializes the accumulator entry from positive zero, writes [epoch] to its mark, and appends
     * the index once in input order. Already touched entries keep their original position. Exact cancellation does
     * not remove support. [epoch] must be nonzero and is caller-managed; marks must be reset before epoch reuse or
     * wrap. The existing touched slice must contain every entry encountered here whose mark already equals [epoch].
     * This is a trusted precondition: the existing touched slice and its marks are not scanned. Capacity is required
     * only for indices whose mark does not yet equal [epoch].
     *
     * Zero [alpha] is evaluated, not treated as a no-op. Thus finite values still become touched and `0 * infinity`
     * produces NaN according to IEEE 754 arithmetic.
     */
    @Suppress("LongParameterList") // parallel input slices plus caller-owned accumulator state
    public fun scatterAxpy(
        alpha: Double,
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        accumulator: DoubleArray,
        marks: IntArray,
        epoch: Int,
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
    ): Int {
        validateScatter(
            indices, indexOffset, values, valueOffset, count,
            accumulator, marks, epoch, touched, touchedOffset, touchedCount,
        )

        return SparseAccumulationKernels.scatterWorkspace(
            alpha, indices, indexOffset, values, valueOffset, count,
            accumulator, marks, epoch, touched, touchedOffset, touchedCount,
        )
    }

    /**
     * Performs [scatterAxpy] while latching arithmetic diagnostics into [arithmeticStatus] at [statusOffset].
     *
     * [ARITHMETIC_NONFINITE] is set when a multiplication or updated accumulator value is nonfinite.
     * [ARITHMETIC_NONZERO_PRODUCT_UNDERFLOW] is set when two nonzero operands produce a zero product. Existing bits in
     * the status element are preserved, allowing one caller-owned element to collect diagnostics across repeated
     * scatters. The IEEE result is always written, and each product and sum is evaluated exactly once.
     */
    @Suppress("LongParameterList") // parallel slices, caller-owned accumulator state, and diagnostic sink
    public fun scatterAxpyChecked(
        alpha: Double,
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        accumulator: DoubleArray,
        marks: IntArray,
        epoch: Int,
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        arithmeticStatus: IntArray,
        statusOffset: Int,
    ): Int {
        requireWindow(arithmeticStatus.size, statusOffset, 1, "arithmetic status")
        requireDistinct(arithmeticStatus, indices, "arithmetic status and indices")
        requireDistinct(arithmeticStatus, marks, "arithmetic status and marks")
        requireDistinct(arithmeticStatus, touched, "arithmetic status and touched")
        validateScatter(
            indices, indexOffset, values, valueOffset, count,
            accumulator, marks, epoch, touched, touchedOffset, touchedCount,
        )

        return SparseAccumulationKernels.scatterWorkspaceChecked(
            alpha, indices, indexOffset, values, valueOffset, count,
            accumulator, marks, epoch, touched, touchedOffset, touchedCount,
            arithmeticStatus, statusOffset, ARITHMETIC_NONFINITE, ARITHMETIC_NONZERO_PRODUCT_UNDERFLOW,
        )
    }

    /**
     * Writes touched accumulator entries in touched order and returns the number written.
     * When [compactExactZeros] is true, both signed zeroes are omitted; NaN and infinities are retained.
     * The accumulator and touched support are left unchanged.
     */
    @Suppress("LongParameterList") // caller-owned source and output slices
    public fun gatherTouched(
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        accumulator: DoubleArray,
        outIndices: IntArray,
        outIndexOffset: Int,
        outValues: DoubleArray,
        outValueOffset: Int,
        compactExactZeros: Boolean = false,
    ): Int {
        validateGather(
            touched,
            touchedOffset,
            touchedCount,
            accumulator,
            outIndices,
            outIndexOffset,
            outValues,
            outValueOffset,
        )
        return SparseAccumulationKernels.gatherWorkspace(
            touched, touchedOffset, touchedCount, accumulator,
            outIndices, outIndexOffset, outValues, outValueOffset,
            compactExactZeros, marks = null,
        )
    }

    /**
     * Writes touched entries as [gatherTouched], then clears every visited accumulator and mark entry.
     * Clearing includes exact zeroes omitted by compaction. The caller resets its touched count to zero after the
     * call; touched storage itself is deliberately left intact for reuse.
     */
    @Suppress("LongParameterList") // caller-owned source, state, and output slices
    public fun gatherClearTouched(
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        accumulator: DoubleArray,
        marks: IntArray,
        outIndices: IntArray,
        outIndexOffset: Int,
        outValues: DoubleArray,
        outValueOffset: Int,
        compactExactZeros: Boolean = false,
    ): Int {
        requireShape(accumulator.size == marks.size) {
            "accumulator and marks lengths differ: ${accumulator.size} vs ${marks.size}"
        }
        validateGather(
            touched,
            touchedOffset,
            touchedCount,
            accumulator,
            outIndices,
            outIndexOffset,
            outValues,
            outValueOffset,
        )
        requireDistinct(marks, touched, "marks and touched")
        requireDistinct(marks, outIndices, "marks and output indices")
        return SparseAccumulationKernels.gatherWorkspace(
            touched, touchedOffset, touchedCount, accumulator,
            outIndices, outIndexOffset, outValues, outValueOffset,
            compactExactZeros, marks,
        )
    }

    /**
     * Clears the selected [values] and [marks] entries without consuming or changing [touched].
     *
     * The two state arrays must have the same logical dimension. The touched window, including an empty window at
     * the array end, and every selected index are validated before mutation. Selected indices must be unique as a
     * caller precondition; verifying uniqueness without temporary storage would require quadratic work. Values are
     * written as positive zero and marks as zero.
     */
    public fun clearTouched(
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        values: DoubleArray,
        marks: IntArray,
    ) {
        requireShape(values.size == marks.size) {
            "values and marks lengths differ: ${values.size} vs ${marks.size}"
        }
        requireWindow(touched.size, touchedOffset, touchedCount, "touched")
        requireDistinct(touched, marks, "touched and marks")
        validateIndices(touched, touchedOffset, touchedCount, values.size, "touched")

        for (k in 0 until touchedCount) {
            val index = touched[touchedOffset + k]
            values[index] = 0.0
            marks[index] = 0
        }
    }

    /**
     * Reduces indexed products into [initial] in strict input order and latches arithmetic diagnostics.
     *
     * Each product is evaluated once, then added to or subtracted from the running result once according to
     * [subtractProducts]. [ARITHMETIC_NONFINITE] reports a nonfinite initial value, operand, product, or updated
     * result. [ARITHMETIC_NONZERO_PRODUCT_UNDERFLOW] reports two nonzero finite operands whose product is zero.
     * Existing status bits are preserved, and the IEEE result is returned rather than converted to an exception.
     * A zero updated sum is not classified as addition underflow because it may be exact cancellation.
     *
     * Status and input windows and every index are validated before any arithmetic is evaluated. Repeated and
     * unsorted indices are permitted and contribute in their supplied order. [values] and [dense] may alias.
     */
    @Suppress("LongParameterList") // two independent input slices plus a caller-owned diagnostic sink
    public fun reduceDotChecked(
        initial: Double,
        subtractProducts: Boolean,
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
        arithmeticStatus: IntArray,
        statusOffset: Int,
    ): Double {
        requireWindow(arithmeticStatus.size, statusOffset, 1, "arithmetic status")
        requireWindow(indices.size, indexOffset, count, "indices")
        requireWindow(values.size, valueOffset, count, "values")
        requireNonoverlap(
            arithmeticStatus,
            statusOffset,
            1,
            indices,
            indexOffset,
            count,
            "arithmetic status and indices",
        )
        validateIndices(indices, indexOffset, count, dense.size, "indices")

        var status = arithmeticStatus[statusOffset]
        var result = initial
        if (!initial.isFinite()) status = status or ARITHMETIC_NONFINITE
        for (k in 0 until count) {
            val left = values[valueOffset + k]
            val right = dense[indices[indexOffset + k]]
            val product = left * right
            val updated = if (subtractProducts) result - product else result + product
            if (!left.isFinite() || !right.isFinite() || !product.isFinite() || !updated.isFinite()) {
                status = status or ARITHMETIC_NONFINITE
            }
            if (left.isFinite() && right.isFinite() && left != 0.0 && right != 0.0 && product == 0.0) {
                status = status or ARITHMETIC_NONZERO_PRODUCT_UNDERFLOW
            }
            result = updated
        }
        arithmeticStatus[statusOffset] = status
        return result
    }

    /**
     * Maximum absolute finite value whose row is active. Inactive entries are ignored, empty active support
     * returns zero, and any active NaN or infinity returns NaN.
     */
    public fun activeColumnMaxAbs(
        rowIndices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        activeRows: BooleanArray,
    ): Double {
        requireWindow(rowIndices.size, indexOffset, count, "row indices")
        requireWindow(values.size, valueOffset, count, "values")
        validateIndices(rowIndices, indexOffset, count, activeRows.size, "row indices")
        return SparseAccumulationKernels.activeMaximum(
            rowIndices,
            indexOffset,
            values,
            valueOffset,
            count,
            activeRows,
        )
    }

    /**
     * Writes relative entry positions for active nonzero entries meeting both caller thresholds.
     *
     * [columnMaximum], [absoluteTolerance], and [relativeThreshold] must be finite and nonnegative. The output
     * reserves [count] positions before filtering. Nonfinite active entries are omitted: a maximum produced by
     * [activeColumnMaxAbs] would already be NaN, so a finite maximum means they are not eligible arithmetic data.
     */
    @Suppress("LongParameterList") // raw sparse slice, activity mask, thresholds, and caller output
    public fun pivotCandidatePositions(
        rowIndices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        activeRows: BooleanArray,
        columnMaximum: Double,
        absoluteTolerance: Double,
        relativeThreshold: Double,
        outPositions: IntArray,
        outOffset: Int,
    ): Int {
        require(columnMaximum.isFinite() && columnMaximum >= 0.0) {
            "column maximum must be finite and nonnegative, got $columnMaximum"
        }
        require(absoluteTolerance.isFinite() && absoluteTolerance >= 0.0) {
            "absolute tolerance must be finite and nonnegative, got $absoluteTolerance"
        }
        require(relativeThreshold.isFinite() && relativeThreshold >= 0.0) {
            "relative threshold must be finite and nonnegative, got $relativeThreshold"
        }
        requireWindow(rowIndices.size, indexOffset, count, "row indices")
        requireWindow(values.size, valueOffset, count, "values")
        requireWindow(outPositions.size, outOffset, count, "candidate output")
        requireNonoverlap(rowIndices, indexOffset, count, outPositions, outOffset, count, "row indices and candidates")
        validateIndices(rowIndices, indexOffset, count, activeRows.size, "row indices")

        return SparseAccumulationKernels.selectPivotCandidates(
            rowIndices, indexOffset, values, valueOffset, count, activeRows,
            absoluteTolerance, relativeThreshold * columnMaximum, outPositions, outOffset,
        )
    }
}

@Suppress("LongParameterList")
private fun validateScatter(
    indices: IntArray,
    indexOffset: Int,
    values: DoubleArray,
    valueOffset: Int,
    count: Int,
    accumulator: DoubleArray,
    marks: IntArray,
    epoch: Int,
    touched: IntArray,
    touchedOffset: Int,
    touchedCount: Int,
) {
    require(epoch != 0) { "scatter epoch must be nonzero" }
    requireShape(accumulator.size == marks.size) {
        "accumulator and marks lengths differ: ${accumulator.size} vs ${marks.size}"
    }
    requireWindow(indices.size, indexOffset, count, "indices")
    requireWindow(values.size, valueOffset, count, "values")
    requireWindow(touched.size, touchedOffset, touchedCount, "touched")
    requireDistinct(accumulator, values, "accumulator and values")
    requireDistinct(marks, indices, "marks and indices")
    requireDistinct(marks, touched, "marks and touched")
    validateIndices(indices, indexOffset, count, accumulator.size, "indices")

    var newTouches = 0
    for (k in 0 until count) {
        if (marks[indices[indexOffset + k]] != epoch) newTouches++
    }
    requireWindow(touched.size, touchedOffset + touchedCount, newTouches, "touched capacity")
    requireNonoverlap(
        indices,
        indexOffset,
        count,
        touched,
        touchedOffset,
        touchedCount + newTouches,
        "indices and touched",
    )
}

@Suppress("LongParameterList")
private fun validateGather(
    touched: IntArray,
    touchedOffset: Int,
    touchedCount: Int,
    accumulator: DoubleArray,
    outIndices: IntArray,
    outIndexOffset: Int,
    outValues: DoubleArray,
    outValueOffset: Int,
) {
    requireWindow(touched.size, touchedOffset, touchedCount, "touched")
    requireWindow(outIndices.size, outIndexOffset, touchedCount, "output indices")
    requireWindow(outValues.size, outValueOffset, touchedCount, "output values")
    requireDistinct(accumulator, outValues, "accumulator and output values")
    requireNonoverlap(
        touched,
        touchedOffset,
        touchedCount,
        outIndices,
        outIndexOffset,
        touchedCount,
        "touched and output indices",
    )
    validateIndices(touched, touchedOffset, touchedCount, accumulator.size, "touched")
}

private fun validateIndices(indices: IntArray, offset: Int, count: Int, dimension: Int, name: String) {
    for (k in 0 until count) {
        val index = indices[offset + k]
        requireIndex(index in 0 until dimension) { "$name entry $index is outside [0, $dimension)" }
    }
}

private fun requireWindow(length: Int, offset: Int, count: Int, name: String) {
    require(offset >= 0 && count >= 0 && offset.toLong() + count <= length) {
        "$name window [$offset, ${offset.toLong() + count}) exceeds length $length"
    }
}

private fun requireDistinct(first: Any, second: Any, name: String) {
    require(first !== second) { "$name must use distinct buffers" }
}

private fun requireNonoverlap(
    first: IntArray,
    firstOffset: Int,
    firstCount: Int,
    second: IntArray,
    secondOffset: Int,
    secondCount: Int,
    name: String,
) {
    require(first !== second || firstOffset + firstCount <= secondOffset || secondOffset + secondCount <= firstOffset) {
        "$name windows must not overlap"
    }
}
