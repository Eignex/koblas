package com.eignex.koblas

/**
 * Mutable column-major storage with unit row stride, shared by [DenseMatrix] and [DenseMatrixView].
 * Entry `A(i, j)` is at `offset + i + j * leadingDimension` in [values].
 *
 * [com.eignex.koblas.dense.DenseBlas] accepts regions in its `gemm`, `syrk`, `trsm`, `gemv` and `trsv`
 * overloads. Other matrix APIs may materialise a region. A region is mutable through its buffer and [set].
 */
public sealed interface DenseMatrixRegion : Matrix {
    /** Shared column-major backing buffer, including entries outside this region. */
    public val values: DoubleArray

    /** Buffer index of the first entry, or an in-bounds buffer boundary for an empty region. */
    public val offset: Int

    /** Distance between consecutive column starts. */
    public val leadingDimension: Int

    /** Writes [v] to `A(i, j)`, checking both logical indices. */
    public operator fun set(i: Int, j: Int, v: Double)

    /**
     * Borrows a rectangle without copying. Mutations through either reference are shared.
     * Offsets and extents are relative to this region; a nested view retains the parent's leading dimension.
     * Empty rectangles at either boundary are valid and access no entries.
     */
    public fun view(
        rowOffset: Int = 0,
        columnOffset: Int = 0,
        rows: Int = this.rows - rowOffset,
        cols: Int = this.cols - columnOffset,
    ): DenseMatrixView {
        requireShape(rowOffset in 0..this.rows && columnOffset in 0..this.cols) { "view origin outside matrix" }
        requireShape(rows >= 0 && rows <= this.rows - rowOffset && cols >= 0 && cols <= this.cols - columnOffset) {
            "view rectangle outside matrix"
        }
        val origin = if (rows == 0 || cols == 0) offset else offset + rowOffset + columnOffset * leadingDimension
        return DenseMatrixView.wrap(rows, cols, values, origin, leadingDimension)
    }
}

/**
 * A borrowed rectangle of a [DoubleArray], with explicit [offset] and [leadingDimension].
 * No entries are copied or owned. The backing buffer can also be mutated by its owner or other views.
 * Unlike [DenseMatrix], [values] need not have exactly `rows * cols` entries. Views use reference equality.
 */
public class DenseMatrixView private constructor(
    override val rows: Int,
    override val cols: Int,
    override val values: DoubleArray,
    override val offset: Int,
    override val leadingDimension: Int,
) : DenseMatrixRegion {
    init {
        requireNonNegativeShape(rows, cols)
        requireShape(offset >= 0 && offset <= values.size) { "view offset outside buffer" }
        requireShape(leadingDimension >= rows) { "leading dimension must be at least the row count" }
        if (rows > 0 && cols > 0) {
            val end = offset.toLong() + (cols - 1).toLong() * leadingDimension + rows
            requireShape(end <= values.size) { "view rectangle exceeds buffer" }
        }
    }

    override fun get(i: Int, j: Int): Double {
        requireInBounds(i, j, rows, cols)
        return values[offset + i + j * leadingDimension]
    }

    override fun set(i: Int, j: Int, v: Double) {
        requireInBounds(i, j, rows, cols)
        values[offset + i + j * leadingDimension] = v
    }

    override fun toArray(): Array<DoubleArray> = Array(rows) { i -> DoubleArray(cols) { j -> get(i, j) } }

    /** Factories for borrowed matrix storage. */
    public companion object {
        /**
         * Borrows [values] without copying. All logical entries must fit, and [leadingDimension] must be
         * at least [rows]. For an empty shape, [offset] can be any boundary in `0..values.size`.
         * Shape, spacing and buffer bounds are validated using overflow-safe arithmetic.
         */
        public fun wrap(
            rows: Int,
            cols: Int,
            values: DoubleArray,
            offset: Int = 0,
            leadingDimension: Int = rows,
        ): DenseMatrixView = DenseMatrixView(rows, cols, values, offset, leadingDimension)
    }
}
