package com.eignex.koblas.core

import com.eignex.koblas.*
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** Read-only matrix contract. Anything that only reads a matrix should take this. */
public interface F64MatrixLike {
    /** Number of rows. */
    public val rows: Int

    /** Number of columns. */
    public val cols: Int

    /** The entry at row (i), column (j). Throws `IndexOutOfBoundsException` outside the shape, whatever the storage. */
    public operator fun get(i: Int, j: Int): Double

    /** Materialise into a fresh `Array<DoubleArray>` of rows, independent of the internal storage. */
    public fun toArray(): Array<DoubleArray>
}

/** The matrix storages koblas itself defines, [F64DenseMatrix] and [F64SparseMatrix]. */
@Serializable
public sealed interface F64MatrixView : F64MatrixLike

/**
 * @property rows the number of rows.
 * @property cols the number of columns.
 * @property data the flat column-major backing of length `rows * cols`, entry `(i, j)` at `i + j * rows`.
 *   The matrix is mutable through this buffer and [set]; do not use it as a hash-map key while mutating it.
 */
@Serializable
@SerialName("F64DenseMatrix")
public class F64DenseMatrix internal constructor(
    override val rows: Int,
    override val cols: Int,
    public val data: DoubleArray,
) : F64MatrixView {

    internal constructor(rows: Int, cols: Int = rows) : this(rows, cols, DoubleArray(entryCount(rows, cols)))

    init {
        requireNonNegativeShape(rows, cols)
        requireShape(data.size.toLong() == rows.toLong() * cols) {
            "data length ${data.size} does not match shape ${rows}x$cols (= ${rows.toLong() * cols})"
        }
    }

    override fun get(i: Int, j: Int): Double {
        requireInBounds(i, j, rows, cols)
        return data[i + j * rows]
    }

    override fun toArray(): Array<DoubleArray> = Array(rows) { i ->
        DoubleArray(cols) { j -> data[i + j * rows] }
    }

    /** Writes (v) at row (i), column (j). */
    public operator fun set(i: Int, j: Int, v: Double) {
        requireInBounds(i, j, rows, cols)
        data[i + j * rows] = v
    }

    /** Offset into [data] where column [j] starts, running contiguously for [rows] entries. */
    internal fun colOffset(j: Int): Int = j * rows

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is F64DenseMatrix) return false
        return rows == other.rows && cols == other.cols && data.contentEquals(other.data)
    }
    override fun hashCode(): Int {
        var h = rows * 31 + cols
        h = 31 * h + data.contentHashCode()
        return h
    }
    override fun toString(): String = "F64DenseMatrix(${rows}x$cols)"

    /** Factories for dense matrices. */
    public companion object {
        /** Copy an `Array<DoubleArray>` of rows into a fresh dense matrix, so rows(i) is row i. */
        public fun of(rows: Array<DoubleArray>): F64DenseMatrix {
            val r = rows.size
            val c = if (r == 0) 0 else rows[0].size
            requireShape(rows.all { it.size == c }) { "all rows must have the same length" }
            val flat = DoubleArray(entryCount(r, c))
            for (i in 0 until r) {
                val row = rows[i]
                for (j in 0 until c) flat[i + j * r] = row[j]
            }
            return F64DenseMatrix(r, c, flat)
        }

        /** Copy an `Array<DoubleArray>` of columns into a fresh dense matrix, so columns(j) is column j. */
        public fun ofColumns(columns: Array<DoubleArray>): F64DenseMatrix {
            val c = columns.size
            val r = if (c == 0) 0 else columns[0].size
            requireShape(columns.all { it.size == r }) { "all columns must have the same length" }
            val flat = DoubleArray(entryCount(r, c))
            for (j in 0 until c) columns[j].copyInto(flat, j * r)
            return F64DenseMatrix(r, c, flat)
        }

        /** Create a zero matrix of shape [rows] x [cols], square when [cols] is omitted. */
        public fun zero(rows: Int, cols: Int = rows): F64DenseMatrix = F64DenseMatrix(rows, cols)

        /** Create a square identity matrix of the given [size], scaled by [diagonal]. */
        public fun diagonal(size: Int, diagonal: Double = 1.0): F64DenseMatrix {
            val m = F64DenseMatrix(size, size)
            for (i in 0 until size) m[i, i] = diagonal
            return m
        }

        /** Create the square matrix whose diagonal is [values]. */
        public fun diagonal(values: DoubleArray): F64DenseMatrix {
            val n = values.size
            val m = F64DenseMatrix(n, n)
            for (i in 0 until n) m.data[i + i * n] = values[i]
            return m
        }

        /** Wrap an existing flat `DoubleArray` without copying. The caller relinquishes ownership. */
        public fun wrap(rows: Int, cols: Int, data: DoubleArray): F64DenseMatrix = F64DenseMatrix(rows, cols, data)

        /** Entry count for a shape, validated first so a negative dimension reports a shape error. */
        private fun entryCount(rows: Int, cols: Int): Int {
            requireNonNegativeShape(rows, cols)
            val count = rows.toLong() * cols
            requireShape(count <= Int.MAX_VALUE) {
                "shape ${rows}x$cols needs $count entries, more than one array can hold"
            }
            return count.toInt()
        }
    }
}
