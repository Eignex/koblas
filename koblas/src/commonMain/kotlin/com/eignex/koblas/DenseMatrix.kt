package com.eignex.koblas

import com.eignex.koblas.internal.storage.*
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.jvm.JvmOverloads
import kotlin.jvm.JvmStatic

/**
 * Mutable column-major matrix, including borrowed rectangles of another matrix or buffer.
 * Entry `A(i, j)` is at `offset + i + j * leadingDimension` in [values]. Factories allocate or borrow
 * storage; [view] shares it. Equality and hashing use logical entries, including signed zeros and NaNs.
 * Serialization writes only the logical rectangle and decodes it into independent compact storage.
 * Do not use the matrix as a hash-map key while mutating its entries.
 *
 * @property rows the number of rows.
 * @property cols the number of columns.
 * @property values the shared backing buffer, which may include entries outside this matrix.
 * @property offset the first logical entry's buffer index, or an in-bounds boundary for an empty matrix.
 * @property leadingDimension the distance between consecutive column starts, at least [rows].
 */
@Serializable(with = DenseMatrixSerializer::class)
@SerialName("DenseMatrix")
public class DenseMatrix internal constructor(
    override val rows: Int,
    override val cols: Int,
    public val values: DoubleArray,
    public val offset: Int = 0,
    public val leadingDimension: Int = rows,
) : MatrixStorage {
    internal constructor(rows: Int, cols: Int = rows) : this(rows, cols, DoubleArray(entryCount(rows, cols)))

    init {
        requireNonNegativeShape(rows, cols)
        requireShape(offset >= 0 && offset <= values.size) { "matrix offset outside buffer" }
        requireShape(leadingDimension >= rows) { "leading dimension must be at least the row count" }
        if (rows > 0 && cols > 0) {
            val end = offset.toLong() + (cols - 1).toLong() * leadingDimension + rows
            requireShape(end <= values.size) { "matrix rectangle exceeds buffer" }
        }
    }

    override fun get(i: Int, j: Int): Double {
        requireInBounds(i, j, rows, cols)
        return values[offset + i + j * leadingDimension]
    }

    override fun toArray(): Array<DoubleArray> = Array(rows) { i ->
        DoubleArray(cols) { j -> values[offset + i + j * leadingDimension] }
    }

    /** Writes (v) at row (i), column (j). */
    override operator fun set(i: Int, j: Int, v: Double) {
        requireInBounds(i, j, rows, cols)
        values[offset + i + j * leadingDimension] = v
    }

    /**
     * Borrows a rectangle without copying. Mutations through either reference are shared.
     * Offsets and extents are relative to this matrix; nested views retain its leading dimension.
     * Empty rectangles at either boundary are valid and access no entries.
     */
    @JvmOverloads
    public fun view(
        rowOffset: Int = 0,
        columnOffset: Int = 0,
        rows: Int = this.rows - rowOffset,
        cols: Int = this.cols - columnOffset,
    ): DenseMatrix {
        requireShape(rowOffset in 0..this.rows && columnOffset in 0..this.cols) { "view origin outside matrix" }
        requireShape(rows >= 0 && rows <= this.rows - rowOffset && cols >= 0 && cols <= this.cols - columnOffset) {
            "view rectangle outside matrix"
        }
        val origin = if (rows == 0 || cols == 0) offset else offset + rowOffset + columnOffset * leadingDimension
        return DenseMatrix(rows, cols, values, origin, leadingDimension)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DenseMatrix) return false
        if (rows != other.rows || cols != other.cols) return false
        if (rows == 0 || cols == 0) return true
        for (j in 0 until cols) {
            for (i in 0 until rows) {
                if (values[offset + i + j * leadingDimension].toBits() !=
                    other.values[other.offset + i + j * other.leadingDimension].toBits()
                ) {
                    return false
                }
            }
        }
        return true
    }
    override fun hashCode(): Int {
        var entriesHash = 1
        if (rows > 0 && cols > 0) {
            for (j in 0 until cols) {
                for (i in 0 until rows) {
                    entriesHash = 31 * entriesHash + values[offset + i + j * leadingDimension].hashCode()
                }
            }
        }
        return (rows * 31 + cols) * 31 + entriesHash
    }
    override fun toString(): String = "DenseMatrix(${rows}x$cols)"

    /** Factories for dense matrices. */
    public companion object {
        /** Copy an `Array<DoubleArray>` of rows into a fresh dense matrix, so rows(i) is row i. */
        @JvmStatic
        public fun ofRows(rows: Array<DoubleArray>): DenseMatrix {
            val r = rows.size
            val c = if (r == 0) 0 else rows[0].size
            requireShape(rows.all { it.size == c }) { "all rows must have the same length" }
            val flat = DoubleArray(entryCount(r, c))
            for (i in 0 until r) {
                val row = rows[i]
                for (j in 0 until c) flat[i + j * r] = row[j]
            }
            return DenseMatrix(r, c, flat)
        }

        /** Create a zero matrix of shape [rows] x [cols], square when [cols] is omitted. */
        @JvmStatic
        @JvmOverloads
        public fun zero(rows: Int, cols: Int = rows): DenseMatrix = DenseMatrix(rows, cols)

        /** Create a square identity matrix of the given [size], scaled by [diagonal]. */
        @JvmStatic
        @JvmOverloads
        public fun diagonal(size: Int, diagonal: Double = 1.0): DenseMatrix {
            val m = DenseMatrix(size, size)
            for (i in 0 until size) m[i, i] = diagonal
            return m
        }

        /**
         * Borrows [values] without copying, with unit row stride and optional column padding.
         * All logical entries must fit and [leadingDimension] must be at least [rows]. For an empty shape,
         * [offset] can be any boundary in `0..values.size`. Bounds use overflow-safe arithmetic.
         */
        @JvmStatic
        @JvmOverloads
        public fun wrap(
            rows: Int,
            cols: Int,
            values: DoubleArray,
            offset: Int = 0,
            leadingDimension: Int = rows,
        ): DenseMatrix = DenseMatrix(rows, cols, values, offset, leadingDimension)

        /** Entry count for a shape, validated first so a negative dimension reports a shape error. */
        internal fun entryCount(rows: Int, cols: Int): Int {
            requireNonNegativeShape(rows, cols)
            val count = rows.toLong() * cols
            requireShape(count <= Int.MAX_VALUE) {
                "shape ${rows}x$cols needs $count entries, more than one array can hold"
            }
            return count.toInt()
        }
    }
}
