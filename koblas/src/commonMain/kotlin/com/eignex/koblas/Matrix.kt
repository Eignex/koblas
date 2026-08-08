package com.eignex.koblas

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Read-only matrix contract: shape, entry access, materialisation. Anything that only reads a matrix should
 * take this.
 *
 * Open, deliberately, and that is the difference from [MatrixView]. koblas cannot know every shape a caller
 * already has — a row-major buffer from another library, a banded or blocked structure, a lazily generated
 * kernel matrix — and requiring a copy into [DenseMatrix] to use any of koblas's routines was a tax on
 * exactly the callers with the most data. Implement this and the read-only routines accept it.
 *
 * The cost of an adapter is that koblas can only reach it through [get]. Its own storages are recognised by
 * type and swept along their contiguous axis; a foreign implementation takes the generic path, which visits
 * `rows × cols` entries through a virtual call. That is the right trade for correctness-with-any-input, and
 * the reason to hand koblas a [DenseMatrix] when the data is dense and hot.
 *
 * @see MatrixView the closed, serializable subset.
 */
interface MatrixLike {
    /** Number of rows. */
    val rows: Int

    /** Number of columns. */
    val cols: Int

    /** Read entry at row [i], column [j]. */
    operator fun get(i: Int, j: Int): Double

    /**
     * Materialise into a fresh `Array<DoubleArray>` of rows. Always
     * allocates; the result is independent of any internal storage.
     */
    fun toArray(): Array<DoubleArray>
}

/**
 * The matrix storages koblas itself defines: [DenseMatrix] and [SparseMatrix], and nothing else ever.
 *
 * Sealed for one reason — serialization. A snapshot has to decode back into the concrete storage it was
 * written from, and a closed set is what lets `kotlinx.serialization` do that without a consumer
 * registering anything. It is also why both storages live in this package: a sealed subtype must be
 * declared alongside its root.
 *
 * Take [MatrixLike] instead unless you genuinely need the closed set. This exists so serialization and
 * exhaustive dispatch work, not to restrict what callers can pass.
 *
 * [get] is `O(1)` on the dense storage and a search within a column on the sparse one, so an algorithm that
 * sweeps every entry through [get] is the wrong shape for a sparse operand — walk the stored entries instead.
 */
@Serializable
sealed interface MatrixView : MatrixLike

/**
 * Dense column-major matrix backed by a single contiguous `DoubleArray` of
 * length `rows * cols`. Element `(i, j)` lives at `data[i + j * rows]`, so
 * each column is a contiguous run of `rows` doubles.
 *
 * Column-major is the layout LAPACK and Fortran define, so a host library
 * reads this backing directly: no row-major wrappers, no transposition into
 * a temporary, and `lda` is simply [rows]. It also matches what the
 * algorithms want. A triangular solve, an eta update and a Householder
 * reflector are all column operations, and [SparseMatrix] is already CSC, so
 * the dense and sparse sides of the library now agree on which axis is
 * contiguous.
 *
 * Flat layout buys three properties: one heap allocation rather than
 * `cols` separate column arrays; cache-friendly sweeps across column
 * boundaries; the SIMD primitives in the internal `Primitives.kt` can stream
 * long runs without re-fetching column references on each iteration.
 *
 * Serializes as its shape plus the flat backing — `{"rows":2,"cols":2,"data":[…]}` — which is both the
 * compact form and the extensible one. A nested `Array<DoubleArray>` of rows would read better and cost a
 * bracket pair per row, but the deciding point is that a bare array cannot carry a polymorphic type
 * discriminator: a `DenseMatrix` has to decode through [MatrixView] as a [SparseMatrix] does. A named-field
 * object carries one, and leaves room to add fields later without breaking readers.
 *
 * Unlike the read-only [MatrixView] contract, the concrete matrix exposes its flat [data] backing and
 * elementwise [set] so in-place algorithms (factorizations, updates) can work without reallocating.
 *
 * @property rows the number of rows.
 * @property cols the number of columns.
 * @property data the flat column-major backing, length `rows * cols`.
 */
@Serializable
@SerialName("DenseMatrix")
class DenseMatrix internal constructor(override val rows: Int, override val cols: Int, val data: DoubleArray) :
    MatrixView {

    constructor(rows: Int, cols: Int = rows) : this(rows, cols, DoubleArray(entryCount(rows, cols)))

    init {
        require(rows >= 0 && cols >= 0) { "negative shape: ${rows}x$cols" }
        require(data.size == rows * cols) {
            "data length ${data.size} does not match shape ${rows}x$cols (= ${rows * cols})"
        }
    }

    override fun get(i: Int, j: Int): Double = data[i + j * rows]
    override fun toArray(): Array<DoubleArray> = Array(rows) { i ->
        DoubleArray(cols) { j -> data[i + j * rows] }
    }

    /** Set entry `(i, j)`. */
    operator fun set(i: Int, j: Int, v: Double) {
        data[i + j * rows] = v
    }

    /** Offset into [data] where column [j] starts; the column runs contiguously for [rows] entries. */
    internal fun colOffset(j: Int): Int = j * rows

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DenseMatrix) return false
        return rows == other.rows && cols == other.cols && data.contentEquals(other.data)
    }
    override fun hashCode(): Int {
        var h = rows * 31 + cols
        h = 31 * h + data.contentHashCode()
        return h
    }
    override fun toString(): String = "DenseMatrix(${rows}x$cols)"

    /** Factory entrypoints for [DenseMatrix]. */
    companion object {
        /** Copy an `Array<DoubleArray>` of rows into a fresh dense matrix. */
        fun of(rows: Array<DoubleArray>): DenseMatrix {
            val r = rows.size
            val c = if (r == 0) 0 else rows[0].size
            require(rows.all { it.size == c }) { "all rows must have the same length" }
            val flat = DoubleArray(r * c)
            // A row of the argument scatters down the columns of the backing, so this transposes as it
            // copies rather than bulk-copying each row.
            for (i in 0 until r) {
                val row = rows[i]
                for (j in 0 until c) flat[i + j * r] = row[j]
            }
            return DenseMatrix(r, c, flat)
        }

        /** Create an NxN identity matrix scaled by [diagonal]. */
        fun diagonal(size: Int, diagonal: Double = 1.0): DenseMatrix {
            val m = DenseMatrix(size, size)
            for (i in 0 until size) m[i, i] = diagonal
            return m
        }

        /** Wrap an existing flat `DoubleArray` of length `rows * cols` without copying. */
        fun wrap(rows: Int, cols: Int, data: DoubleArray): DenseMatrix = DenseMatrix(rows, cols, data)

        /**
         * Entry count for a shape, validated first: computing `rows * cols` for the backing would
         * otherwise allocate a negative-length array and fail with `NegativeArraySizeException` before
         * the constructor's own shape check could report the problem.
         */
        private fun entryCount(rows: Int, cols: Int): Int {
            require(rows >= 0 && cols >= 0) { "negative shape: ${rows}x$cols" }
            return rows * cols
        }
    }
}
