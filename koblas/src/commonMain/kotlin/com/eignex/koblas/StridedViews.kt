package com.eignex.koblas

import com.eignex.koblas.requireInBounds
import com.eignex.koblas.requireNonNegativeShape
import com.eignex.koblas.requireShape

/**
 * A mutable live view of [size] entries in [data], starting at [offset] and separated by [stride].
 * Negative stride is supported when both ends remain in the buffer. The view never copies [data], so mutations
 * through the view or any other reference to the array are visible to each other.
 */
public class StridedVectorView(
    public val data: DoubleArray,
    public val offset: Int,
    override val size: Int,
    public val stride: Int = 1,
) : VectorLike {
    init {
        requireShape(size >= 0) { "negative size: $size" }
        require(stride != 0) { "stride must not be zero" }
        requireViewBounds(data.size, offset, size, stride, "vector")
    }

    override fun get(i: Int): Double {
        requireInBounds(i, size)
        return data[offset + i * stride]
    }

    /** Writes [value] at logical index [i]. */
    public operator fun set(i: Int, value: Double) {
        requireInBounds(i, size)
        data[offset + i * stride] = value
    }

    override fun toDoubleArray(): DoubleArray = DoubleArray(size) { get(it) }

    override fun toString(): String = "StridedVectorView(size=$size, offset=$offset, stride=$stride)"
}

/**
 * A mutable live column-major matrix view. Entry `(i, j)` is at
 * `offset + i + j * leadingDimension` in [data], so panels can retain their parent's physical column stride.
 * The view never copies [data], so mutations through the view or any other reference to the array are visible
 * to each other.
 */
public class StridedMatrixView(
    override val rows: Int,
    override val cols: Int,
    public val data: DoubleArray,
    /** Physical buffer index of entry `(0, 0)`. */
    public val offset: Int = 0,
    /** Physical distance between the starts of adjacent columns. */
    public val leadingDimension: Int = maxOf(1, rows),
) : MatrixLike {
    init {
        requireNonNegativeShape(rows, cols)
        require(leadingDimension >= maxOf(1, rows)) {
            "leadingDimension $leadingDimension is smaller than max(1, rows) ${maxOf(1, rows)}"
        }
        val span = if (rows == 0 || cols == 0) 0 else (cols - 1).toLong() * leadingDimension + rows
        requireShape(offset >= 0 && offset.toLong() + span <= data.size) {
            "matrix view offset $offset and span $span exceed buffer length ${data.size}"
        }
    }

    override fun get(i: Int, j: Int): Double {
        requireInBounds(i, j, rows, cols)
        return data[offset + i + j * leadingDimension]
    }

    /** Writes [value] at row [i], column [j]. */
    public operator fun set(i: Int, j: Int, value: Double) {
        requireInBounds(i, j, rows, cols)
        data[offset + i + j * leadingDimension] = value
    }

    override fun toArray(): Array<DoubleArray> = Array(rows) { i -> DoubleArray(cols) { j -> get(i, j) } }

    /** A live submatrix retaining this view's physical [leadingDimension]. */
    public fun view(row: Int, rows: Int, column: Int, cols: Int): StridedMatrixView {
        requireIndex(row in 0..this.rows) { "row $row is outside [0, ${this.rows}]" }
        requireIndex(column in 0..this.cols) { "column $column is outside [0, ${this.cols}]" }
        requireShape(rows >= 0 && row.toLong() + rows <= this.rows) {
            "row range [$row, ${row.toLong() + rows}) exceeds $this"
        }
        requireShape(cols >= 0 && column.toLong() + cols <= this.cols) {
            "column range [$column, ${column.toLong() + cols}) exceeds $this"
        }
        return StridedMatrixView(
            rows,
            cols,
            data,
            sliceOffset(row, column),
            leadingDimension,
        )
    }

    /** Live column [j], contiguous even when this view is a panel. */
    public fun column(j: Int): StridedVectorView {
        requireInBounds(j, cols)
        return StridedVectorView(data, sliceOffset(0, j), rows)
    }

    /** Live row [i], strided by [leadingDimension]. */
    public fun row(i: Int): StridedVectorView {
        requireInBounds(i, rows)
        return StridedVectorView(data, sliceOffset(i, 0), cols, leadingDimension)
    }

    private fun sliceOffset(row: Int, column: Int): Int {
        // Empty boundary slices have no physical entries; their logical origin can lie beyond the buffer,
        // especially when the final column has no trailing padding. Keep their sentinel offset in bounds.
        val origin = offset.toLong() + row + column.toLong() * leadingDimension
        return minOf(origin, data.size.toLong()).toInt()
    }

    /**
     * Whether this matrix and [other] address at least one common buffer entry.
     *
     * Every strided product checks this before it runs, so the cheap answers come first: a different buffer,
     * then physical spans that do not meet, then, where the two lie on one grid of columns, a rectangle
     * intersection. Only views whose leading dimensions differ, or whose columns wrap across that grid, are
     * walked entry by entry.
     */
    public fun overlaps(other: StridedMatrixView): Boolean {
        if (data !== other.data || physicalSpan == 0 || other.physicalSpan == 0) return false
        if (offset + physicalSpan <= other.offset || other.offset + other.physicalSpan <= offset) return false
        if (leadingDimension == other.leadingDimension) {
            val origin = minOf(offset, other.offset)
            val here = offset - origin
            val there = other.offset - origin
            val row = here % leadingDimension
            val otherRow = there % leadingDimension
            // A column that runs past the end of its grid column continues in the next one, which is not a
            // rectangle any more, so those fall through to the walk below.
            if (row + rows <= leadingDimension && otherRow + other.rows <= leadingDimension) {
                val column = here / leadingDimension
                val otherColumn = there / leadingDimension
                return row < otherRow + other.rows && otherRow < row + rows &&
                    column < otherColumn + other.cols && otherColumn < column + cols
            }
        }
        val first = if (rows.toLong() * cols <= other.rows.toLong() * other.cols) this else other
        val second = if (first === this) other else this
        for (j in 0 until first.cols) {
            for (i in 0 until first.rows) {
                if (second.containsPhysicalIndex(first.offset + i + j * first.leadingDimension)) return true
            }
        }
        return false
    }

    /** Whether this matrix and [other] address at least one common buffer entry. */
    public fun overlaps(other: StridedVectorView): Boolean {
        if (data !== other.data || physicalSpan == 0 || other.size == 0) return false
        val reach = (other.size - 1).toLong() * other.stride
        val low = if (other.stride > 0) other.offset.toLong() else other.offset + reach
        val high = if (other.stride > 0) other.offset + reach else other.offset.toLong()
        if (offset + physicalSpan <= low || high < offset) return false
        for (i in 0 until other.size) if (containsPhysicalIndex(other.offset + i * other.stride)) return true
        return false
    }

    /** Buffer entries from [offset] to the last one this view can reach, the last included. */
    private val physicalSpan: Int
        get() = if (rows == 0 || cols == 0) 0 else (cols - 1) * leadingDimension + rows

    private fun containsPhysicalIndex(index: Int): Boolean {
        val relative = index - offset
        if (relative < 0) return false
        val column = relative / leadingDimension
        val row = relative % leadingDimension
        return column < cols && row < rows
    }

    override fun toString(): String =
        "StridedMatrixView(${rows}x$cols, offset=$offset, leadingDimension=$leadingDimension)"
}

/** Whether these vectors address at least one common buffer entry. */
public fun StridedVectorView.overlaps(other: StridedVectorView): Boolean {
    if (data !== other.data) return false
    val first = if (size <= other.size) this else other
    val second = if (first === this) other else this
    for (i in 0 until first.size) {
        val physical = first.offset + i * first.stride
        val relative = physical - second.offset
        if (relative % second.stride == 0 && relative / second.stride in 0 until second.size) return true
    }
    return false
}

/** A borrowed view over this entire owned matrix. */
public fun DenseMatrix.asView(): StridedMatrixView = StridedMatrixView(rows, cols, data)

/** A borrowed panel of this owned matrix. */
public fun DenseMatrix.view(row: Int, rows: Int, column: Int, cols: Int): StridedMatrixView =
    asView().view(row, rows, column, cols)

/** A borrowed view over this entire owned vector. */
public fun DenseVector.asView(): StridedVectorView = StridedVectorView(data, 0, size)

/** A borrowed strided slice of this owned vector. */
public fun DenseVector.view(offset: Int, size: Int, stride: Int = 1): StridedVectorView =
    StridedVectorView(data, offset, size, stride)

private fun requireViewBounds(bufferSize: Int, offset: Int, size: Int, stride: Int, description: String) {
    if (size == 0) {
        requireShape(offset in 0..bufferSize) { "$description view offset $offset exceeds buffer length $bufferSize" }
        return
    }
    val last = offset.toLong() + (size - 1).toLong() * stride
    val firstPhysical = minOf(offset.toLong(), last)
    val lastPhysical = maxOf(offset.toLong(), last)
    requireShape(firstPhysical >= 0 && lastPhysical < bufferSize) {
        "$description view addresses [$firstPhysical, $lastPhysical] outside buffer length $bufferSize"
    }
}
