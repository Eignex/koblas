package com.eignex.koblas.sparse

import com.eignex.koblas.ExperimentalKoblasApi
import kotlin.math.abs

/**
 * Allocation-free sparse arithmetic over caller-owned array slices and workspaces.
 *
 * The caller owns every buffer and retains it between calls. Input indices and touched indices must be unique
 * within their supplied slices. That uniqueness, and agreement between an active mark and the touched slice, are
 * caller preconditions: checking either without another workspace would require quadratic work or a full-dimension
 * scan. Bounds, windows, capacities, overlap, and scatter's incoming indices are validated before any destination
 * is mutated. The existing touched slice is trusted and is never scanned by scatter. Roles without two explicit
 * comparable slices must use distinct arrays; explicit
 * `IntArray` slices may share an array only when their reserved windows do not overlap.
 *
 * These helpers manipulate arithmetic values and active support only. Pivot selection, merit calculations,
 * permutations, dropping policy, and factorization state remain caller responsibilities.
 */
@ExperimentalKoblasApi
public object SparseWorkspace {
    /** Bit reported by [scatterAxpyChecked] when a product or updated accumulator value is not finite. */
    public const val SCATTER_NONFINITE: Int = 1

    /** Bit reported by [scatterAxpyChecked] when nonzero operands produce a zero product. */
    public const val SCATTER_NONZERO_PRODUCT_UNDERFLOW: Int = 2

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

        var total = touchedCount
        for (k in 0 until count) {
            val index = indices[indexOffset + k]
            if (marks[index] != epoch) {
                accumulator[index] = 0.0
                marks[index] = epoch
                touched[touchedOffset + total] = index
                total++
            }
            accumulator[index] += alpha * values[valueOffset + k]
        }
        return total
    }

    /**
     * Performs [scatterAxpy] while latching arithmetic diagnostics into [arithmeticStatus] at [statusOffset].
     *
     * [SCATTER_NONFINITE] is set when a multiplication or updated accumulator value is nonfinite.
     * [SCATTER_NONZERO_PRODUCT_UNDERFLOW] is set when two nonzero operands produce a zero product. Existing bits in
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

        var flags = arithmeticStatus[statusOffset]
        var total = touchedCount
        for (k in 0 until count) {
            val index = indices[indexOffset + k]
            if (marks[index] != epoch) {
                accumulator[index] = 0.0
                marks[index] = epoch
                touched[touchedOffset + total] = index
                total++
            }
            val value = values[valueOffset + k]
            val product = alpha * value
            val updated = accumulator[index] + product
            if (!product.isFinite() || !updated.isFinite()) flags = flags or SCATTER_NONFINITE
            if (alpha != 0.0 && value != 0.0 && product == 0.0) {
                flags = flags or SCATTER_NONZERO_PRODUCT_UNDERFLOW
            }
            accumulator[index] = updated
        }
        arithmeticStatus[statusOffset] = flags
        return total
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
        return gather(
            touched, touchedOffset, touchedCount, accumulator,
            outIndices, outIndexOffset, outValues, outValueOffset,
            compactExactZeros, clear = false, marks = null,
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
        require(accumulator.size == marks.size) {
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
        return gather(
            touched, touchedOffset, touchedCount, accumulator,
            outIndices, outIndexOffset, outValues, outValueOffset,
            compactExactZeros, clear = true, marks = marks,
        )
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
        var maximum = 0.0
        for (k in 0 until count) {
            if (!activeRows[rowIndices[indexOffset + k]]) continue
            val value = values[valueOffset + k]
            if (!value.isFinite()) return Double.NaN
            maximum = maxOf(maximum, abs(value))
        }
        return maximum
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

        val relativeCutoff = relativeThreshold * columnMaximum
        var written = 0
        for (k in 0 until count) {
            if (!activeRows[rowIndices[indexOffset + k]]) continue
            val value = values[valueOffset + k]
            if (!value.isFinite() || value == 0.0) continue
            val magnitude = abs(value)
            if (magnitude >= absoluteTolerance && magnitude >= relativeCutoff) {
                outPositions[outOffset + written] = k
                written++
            }
        }
        return written
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
    require(accumulator.size == marks.size) {
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

@Suppress("LongParameterList")
private fun gather(
    touched: IntArray,
    touchedOffset: Int,
    touchedCount: Int,
    accumulator: DoubleArray,
    outIndices: IntArray,
    outIndexOffset: Int,
    outValues: DoubleArray,
    outValueOffset: Int,
    compactExactZeros: Boolean,
    clear: Boolean,
    marks: IntArray?,
): Int {
    var written = 0
    for (k in 0 until touchedCount) {
        val index = touched[touchedOffset + k]
        val value = accumulator[index]
        if (!compactExactZeros || value != 0.0) {
            outIndices[outIndexOffset + written] = index
            outValues[outValueOffset + written] = value
            written++
        }
        if (clear) {
            accumulator[index] = 0.0
            marks!![index] = 0
        }
    }
    return written
}

private fun validateIndices(indices: IntArray, offset: Int, count: Int, dimension: Int, name: String) {
    for (k in 0 until count) {
        val index = indices[offset + k]
        require(index in 0 until dimension) { "$name entry $index is outside [0, $dimension)" }
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
