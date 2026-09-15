package com.eignex.koblas.sparse

import kotlin.jvm.JvmStatic
import kotlin.math.abs

/**
 * The generic sparse primitives, over storage the caller owns and keeps between calls.
 *
 * These are numerical leaves, not algorithms. A consumer that owns a factorization builds its own workflows on
 * them; what lives here is the arithmetic those workflows share, with its semantics stated rather than implied.
 *
 * Every entry point is allocation-free and writes only through the slices it is handed. An accumulator is a
 * dense scratch array paired with a mark array and an epoch, so a column is touched without clearing the whole
 * dimension: a mark that does not equal the current epoch means the accumulator entry is stale and is
 * overwritten rather than added to. The touched list records which entries an epoch reached, so clearing costs
 * what was written rather than the dimension. An epoch is never zero, because a cleared mark is zero and an
 * epoch that collided with it would read every cleared entry as live.
 *
 * Scatter input indices, and touched indices used by a gather, must be unique within their slices; reductions
 * explicitly permit repeated indices. Uniqueness is a caller precondition, because checking it without
 * temporary storage would need quadratic work or a full-dimension scan. A stored exact zero is structural and
 * survives unless a caller asks for compaction. Nonfinite values propagate as the arithmetic produces them,
 * except where a checked entry point reports them instead.
 */
public object SparsePrimitives {

    /**
     * Accumulates `alpha * values` into [accumulator] at [indices], returning the new touched count.
     *
     * An index whose mark is not [epoch] is treated as absent: the accumulator entry is set to positive zero,
     * marked, and appended to [touched] before the contribution is added. So a caller reuses one dense
     * accumulator across columns without clearing it, and [touched] names exactly what to clear afterwards.
     *
     * [epoch] must not be zero, which is the value a cleared mark holds.
     */
    @Suppress("LongParameterList")
    @JvmStatic
    public fun scatterWorkspace(
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
        require(epoch != 0) { "scatter epoch must be nonzero" }
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
     * [scatterWorkspace] that also reports nonfinite arithmetic and product underflow.
     *
     * The detection is the reason this exists separately: a solver deciding whether a pivot is usable needs to
     * know that a contribution went nonfinite or that a product underflowed to zero, and ordinary BLAS has no
     * way to say so. Findings are OR-ed into [arithmeticStatus] at [statusOffset] rather than thrown, so one
     * pass over a column reports everything it saw and the caller decides what it means.
     */
    @Suppress("LongParameterList")
    @JvmStatic
    public fun scatterWorkspaceChecked(
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
        nonfiniteFlag: Int,
        underflowFlag: Int,
    ): Int {
        require(epoch != 0) { "scatter epoch must be nonzero" }
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
            if (!product.isFinite() || !updated.isFinite()) flags = flags or nonfiniteFlag
            if (alpha != 0.0 && value != 0.0 && product == 0.0) flags = flags or underflowFlag
            accumulator[index] = updated
        }
        arithmeticStatus[statusOffset] = flags
        return total
    }

    /**
     * Reads the entries [touched] names out of [accumulator] into an output support, returning how many were
     * written.
     *
     * Emission follows [touched], which is the order entries were first reached and not necessarily ascending;
     * a caller needing ascending order sorts afterwards. [compactExactZeros] drops entries that are exactly
     * zero, which is the one place a structural zero is removed and is why it is a parameter rather than a
     * rule. Passing [marks] clears each mark as its entry is read, so the accumulator is left reusable in the
     * same pass.
     */
    @Suppress("LongParameterList")
    @JvmStatic
    public fun gatherWorkspace(
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        accumulator: DoubleArray,
        outIndices: IntArray,
        outIndexOffset: Int,
        outValues: DoubleArray,
        outValueOffset: Int,
        compactExactZeros: Boolean,
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
            if (marks != null) {
                accumulator[index] = 0.0
                marks[index] = 0
            }
        }
        return written
    }

    /**
     * Returns the entries [touched] names to their cleared state, leaving [touched] itself alone.
     *
     * [gatherWorkspace] clears as it reads, which serves a caller that wants the column's values. This serves
     * one that does not: a column abandoned partway, or one whose result was written somewhere else. Either
     * way the accumulator has to be reusable before the next epoch, and clearing what an epoch reached costs
     * what was written rather than the dimension.
     *
     * An accumulator entry is written as positive zero and its mark as zero, which is the value a scatter
     * epoch is forbidden to take.
     */
    @JvmStatic
    public fun clearWorkspace(
        touched: IntArray,
        touchedOffset: Int,
        touchedCount: Int,
        accumulator: DoubleArray,
        marks: IntArray,
    ) {
        for (k in 0 until touchedCount) {
            val index = touched[touchedOffset + k]
            accumulator[index] = 0.0
            marks[index] = 0
        }
    }

    /**
     * The largest magnitude among the entries whose row is still active, or zero when none is.
     *
     * The mask is what makes this a primitive rather than a plain reduction: a solver eliminates rows as it
     * goes, and the maximum it needs is over what remains. An empty selection returns zero rather than naming a
     * position, so a column with nothing active is distinguishable from one whose first entry won. A nonfinite
     * entry returns NaN immediately, because a magnitude comparison against it would answer arbitrarily.
     */
    @Suppress("LongParameterList")
    @JvmStatic
    public fun activeMaximum(
        rowIndices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        activeRows: BooleanArray,
    ): Double {
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
     * Writes the positions within a column that clear both tolerances, returning how many were written.
     *
     * Positions, not row indices, so the caller can read the value and the row back from its own slices without
     * a second search. An entry that is nonfinite or exactly zero is never a candidate. This reports what is
     * eligible; which candidate to take, and what the tolerances should be, is pivot policy and belongs to the
     * consumer that owns the factorization.
     *
     * Both tolerances must be finite and nonnegative, and are rejected rather than applied. Scaling
     * [activeMaximum] by a relative threshold is the natural way to reach [relativeCutoff], and that yields NaN
     * for a column holding a nonfinite entry. Every comparison against NaN is false, so admitting one would
     * report an empty candidate set and a numerically broken column would read as merely unpivotable.
     */
    @Suppress("LongParameterList")
    @JvmStatic
    public fun selectPivotCandidates(
        rowIndices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        activeRows: BooleanArray,
        absoluteTolerance: Double,
        relativeCutoff: Double,
        outPositions: IntArray,
        outOffset: Int,
    ): Int {
        require(absoluteTolerance.isFinite() && absoluteTolerance >= 0.0) {
            "absolute tolerance must be finite and nonnegative, got $absoluteTolerance"
        }
        require(relativeCutoff.isFinite() && relativeCutoff >= 0.0) {
            "relative cutoff must be finite and nonnegative, got $relativeCutoff"
        }
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
