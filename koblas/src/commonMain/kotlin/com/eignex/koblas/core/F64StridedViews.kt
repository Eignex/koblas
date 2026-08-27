package com.eignex.koblas.core

import com.eignex.koblas.requireInBounds
import com.eignex.koblas.requireNonNegativeShape
import com.eignex.koblas.requireShape

/** Whether a dense storage object owns its buffer or borrows it from another object. */
public enum class BufferOwnership {
    /** The storage owns the buffer passed to it; callers must not independently mutate it. */
    OWNED,

    /** The storage is a live view; the source owner and every overlapping view observe mutations. */
    BORROWED,
}

/**
 * A mutable borrowed view of [size] entries in [data], starting at [offset] and separated by [stride].
 * Negative stride is supported when both ends remain in the buffer. The view never copies or owns [data].
 */
public class F64StridedVectorView(
    public val data: DoubleArray,
    public val offset: Int,
    override val size: Int,
    public val stride: Int = 1,
) : F64VectorLike {
    /** This view borrows [data]. */
    public val ownership: BufferOwnership get() = BufferOwnership.BORROWED

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

    override fun toString(): String = "F64StridedVectorView(size=$size, offset=$offset, stride=$stride)"
}

/**
 * A mutable borrowed column-major matrix view. Entry `(i, j)` is at
 * `offset + i + j * leadingDimension` in [data], so panels can retain their parent's physical column stride.
 */
public class F64StridedMatrixView(
    override val rows: Int,
    override val cols: Int,
    public val data: DoubleArray,
    /** Physical buffer index of entry `(0, 0)`. */
    public val offset: Int = 0,
    /** Physical distance between the starts of adjacent columns. */
    public val leadingDimension: Int = maxOf(1, rows),
) : F64MatrixLike {
    /** This view borrows [data]. */
    public val ownership: BufferOwnership get() = BufferOwnership.BORROWED

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
    public fun view(row: Int, rows: Int, column: Int, cols: Int): F64StridedMatrixView {
        requireShape(row >= 0 && rows >= 0 && row.toLong() + rows <= this.rows) {
            "row range [$row, ${row.toLong() + rows}) exceeds $this"
        }
        requireShape(column >= 0 && cols >= 0 && column.toLong() + cols <= this.cols) {
            "column range [$column, ${column.toLong() + cols}) exceeds $this"
        }
        return F64StridedMatrixView(
            rows,
            cols,
            data,
            offset + row + column * leadingDimension,
            leadingDimension,
        )
    }

    /** Live column [j], contiguous even when this view is a panel. */
    public fun column(j: Int): F64StridedVectorView {
        requireInBounds(j, cols)
        return F64StridedVectorView(data, offset + j * leadingDimension, rows)
    }

    /** Live row [i], strided by [leadingDimension]. */
    public fun row(i: Int): F64StridedVectorView {
        requireInBounds(i, rows)
        return F64StridedVectorView(data, offset + i, cols, leadingDimension)
    }

    /** Whether this matrix and [other] address at least one common buffer entry. */
    public fun overlaps(other: F64StridedMatrixView): Boolean {
        if (data !== other.data) return false
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
    public fun overlaps(other: F64StridedVectorView): Boolean {
        if (data !== other.data) return false
        for (i in 0 until other.size) if (containsPhysicalIndex(other.offset + i * other.stride)) return true
        return false
    }

    private fun containsPhysicalIndex(index: Int): Boolean {
        val relative = index - offset
        if (relative < 0) return false
        val column = relative / leadingDimension
        val row = relative % leadingDimension
        return column < cols && row < rows
    }

    override fun toString(): String =
        "F64StridedMatrixView(${rows}x$cols, offset=$offset, leadingDimension=$leadingDimension)"
}

/** Whether these vectors address at least one common buffer entry. */
public fun F64StridedVectorView.overlaps(other: F64StridedVectorView): Boolean {
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
public fun F64DenseMatrix.asView(): F64StridedMatrixView = F64StridedMatrixView(rows, cols, data)

/** A borrowed panel of this owned matrix. */
public fun F64DenseMatrix.view(row: Int, rows: Int, column: Int, cols: Int): F64StridedMatrixView =
    asView().view(row, rows, column, cols)

/** A borrowed view over this entire owned vector. */
public fun F64DenseVector.asView(): F64StridedVectorView = F64StridedVectorView(data, 0, size)

/** A borrowed strided slice of this owned vector. */
public fun F64DenseVector.view(offset: Int, size: Int, stride: Int = 1): F64StridedVectorView =
    F64StridedVectorView(data, offset, size, stride)

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
