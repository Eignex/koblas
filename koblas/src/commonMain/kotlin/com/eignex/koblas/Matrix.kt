package com.eignex.koblas

/**
 * A dense, row-major matrix backed by a single contiguous [data] array of length `rows * cols`: element
 * `(i, j)` is `data[i * cols + j]`. The flat layout is what BLAS/LAPACK backends consume directly, so a
 * real-backend [LinearAlgebra] can pass [data] across the FFI boundary without repacking.
 *
 * @property rows the number of rows.
 * @property cols the number of columns.
 * @property data the row-major elements, length `rows * cols`.
 */
class Matrix(val rows: Int, val cols: Int, val data: DoubleArray) {
    init {
        require(rows >= 0 && cols >= 0) { "negative dimensions ${rows}x$cols" }
        require(data.size == rows * cols) { "data length ${data.size} != $rows*$cols" }
    }

    /** Element `(i, j)`. */
    operator fun get(i: Int, j: Int): Double = data[i * cols + j]

    /** Set element `(i, j)`. */
    operator fun set(i: Int, j: Int, value: Double) {
        data[i * cols + j] = value
    }

    override fun equals(other: Any?): Boolean = this === other || (
        other is Matrix && rows == other.rows && cols == other.cols && data.contentEquals(
            other.data,
        )
        )

    override fun hashCode(): Int = 31 * (31 * rows + cols) + data.contentHashCode()

    override fun toString(): String = "Matrix(${rows}x$cols)"

    /** Factories for common matrix shapes. */
    companion object {
        /** A `rows x cols` matrix of zeros. */
        fun zeros(rows: Int, cols: Int = rows): Matrix = Matrix(rows, cols, DoubleArray(rows * cols))

        /** The `n x n` identity matrix. */
        fun identity(n: Int): Matrix = Matrix(n, n, DoubleArray(n * n)).also { for (i in 0 until n) it[i, i] = 1.0 }

        /** A matrix from row vectors; every row must have the same length. */
        fun ofRows(vararg rows: DoubleArray): Matrix {
            val r = rows.size
            val c = if (r == 0) 0 else rows[0].size
            val data = DoubleArray(r * c)
            for (i in 0 until r) {
                require(rows[i].size == c) { "ragged rows: ${rows[i].size} != $c" }
                rows[i].copyInto(data, i * c)
            }
            return Matrix(r, c, data)
        }
    }
}
