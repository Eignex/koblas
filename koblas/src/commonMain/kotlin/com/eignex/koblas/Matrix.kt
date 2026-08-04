package com.eignex.koblas

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ArraySerializer
import kotlinx.serialization.builtins.DoubleArraySerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/**
 * Read-only N-by-M matrix. Sealed alongside [VectorView] so snapshots
 * round-trip through `kotlinx.serialization` with their concrete storage
 * preserved. Public surface is read-only; shape, entry access, materialise
 * to `Array<DoubleArray>`. Arithmetic (`gemv`, `ger`, the Cholesky
 * suite, …) lives as free functions over the views; mutation is `internal`.
 *
 * [DenseMatrix] and [SparseMatrix] are the two storages. Being sealed is why
 * both live in this package rather than moving with the operations that consume
 * them: a sealed subtype has to be declared alongside its root, and the closed
 * set is what makes the polymorphic serialization work without a consumer
 * registering anything.
 *
 * [get] is `O(1)` on the dense storage and a search within a column on the
 * sparse one, so an algorithm that sweeps every entry through [get] is the wrong
 * shape for a sparse operand — walk the stored entries instead.
 */
@Serializable
sealed interface MatrixView {
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
 * The on-the-wire form is a 2D `Array<DoubleArray>` of rows, for readability
 * when inspecting JSON / CBOR payloads, and is unaffected by the storage
 * order. The custom [DenseMatrixSerializer] bridges the two; encoding writes
 * rows, decoding reads them back and packs them into the flat backing.
 *
 * Unlike the read-only [MatrixView] contract, the concrete matrix exposes its flat [data] backing and
 * elementwise [set] so in-place algorithms (factorizations, updates) can work without reallocating.
 *
 * @property rows the number of rows.
 * @property cols the number of columns.
 * @property data the flat column-major backing, length `rows * cols`.
 */
@Serializable(with = DenseMatrixSerializer::class)
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

/** Serialises [DenseMatrix] as a 2D `Array<DoubleArray>` of rows. The flat in-memory backing is an
 *  implementation detail; the wire shape stays stable across layout changes, so payloads written before
 *  the move to column-major storage still decode correctly. A zero-row matrix encodes
 *  to `[]`, so its column count is not recoverable — a `0×N` matrix decodes as `0×0` (a degenerate case
 *  the format does not distinguish). */
@OptIn(ExperimentalSerializationApi::class)
internal object DenseMatrixSerializer : KSerializer<DenseMatrix> {
    private val inner = ArraySerializer(DoubleArraySerializer())
    override val descriptor: SerialDescriptor get() = inner.descriptor
    override fun serialize(encoder: Encoder, value: DenseMatrix) =
        encoder.encodeSerializableValue(inner, value.toArray())
    override fun deserialize(decoder: Decoder): DenseMatrix = DenseMatrix.of(decoder.decodeSerializableValue(inner))
}
