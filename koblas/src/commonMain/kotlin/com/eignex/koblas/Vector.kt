package com.eignex.koblas

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Read-only N-vector with sealed dense / sparse backing. Callers see the
 * same surface either way; query the size, read entries by index,
 * materialise to a `DoubleArray`.
 *
 * The split between [DenseVector] and [SparseVector] is purely a backing-
 * storage choice: dense pays per coordinate, sparse pays per nonzero. Most
 * code iterates via the [forEachStored] extension to walk only the populated
 * entries; sparse callers feed sparse vectors without the cost of dense
 * materialisation, and dense callers walk every index the same way.
 *
 * The view surface is read-only; arithmetic (`dot`, `axpy`, `scale`, …) lives
 * as free functions over the views, and mutation is `internal`. For ad-hoc math
 * reach for the `DoubleArray` materialisation via [toDoubleArray].
 *
 * Subtypes are sealed and `@Serializable` so snapshots round-trip through
 * `kotlinx.serialization` with their concrete storage preserved.
 */
@Serializable
sealed interface VectorView {
    /** Number of entries (including stored zeros for sparse). */
    val size: Int

    /**
     * Read entry at [i]. O(1) for [DenseVector], O(nnz) linear scan for
     * [SparseVector]. Use the internal `forEachStored` extension when you
     * want to walk the populated entries without per-index lookup cost.
     */
    operator fun get(i: Int): Double

    /**
     * Materialise into a fresh dense `DoubleArray`. Always allocates;
     * the returned array is independent of any internal storage, so the
     * caller is free to mutate it.
     */
    fun toDoubleArray(): DoubleArray
}

/**
 * Dense double-precision vector backed by a flat `DoubleArray`. The
 * default carrier when the caller already has a dense array or expects
 * most entries to be populated.
 *
 * Construction goes through the [Companion] factories: [DenseVector.of]
 * (copy a `DoubleArray`) or [DenseVector.zero] (allocate a zero vector
 * of given size). Unlike the read-only [VectorView] contract, the concrete
 * vector exposes its [data] backing and elementwise [set] for in-place updates.
 *
 * @property data the flat backing array.
 */
@Serializable
@SerialName("DenseVector")
class DenseVector internal constructor(val data: DoubleArray) : VectorView {

    constructor(size: Int) : this(DoubleArray(size))

    override val size: Int get() = data.size
    override fun get(i: Int): Double = data[i]
    override fun toDoubleArray(): DoubleArray = data.copyOf()

    /** Set entry [i]. */
    operator fun set(i: Int, v: Double) {
        data[i] = v
    }

    override fun equals(other: Any?): Boolean =
        this === other || (other is DenseVector && data.contentEquals(other.data))
    override fun hashCode(): Int = data.contentHashCode()
    override fun toString(): String = "DenseVector(size=$size)"

    /** Factory entrypoints for [DenseVector]. */
    companion object {
        /** Copy a `DoubleArray` into a fresh dense vector. */
        fun of(values: DoubleArray): DenseVector = DenseVector(values.copyOf())

        /** Create a zero vector of length [size]. */
        fun zero(size: Int): DenseVector = DenseVector(size)

        /** Wrap an existing `DoubleArray` without copying. Caller relinquishes ownership. */
        fun wrap(data: DoubleArray): DenseVector = DenseVector(data)
    }
}

/**
 * Compressed sparse vector: parallel [indices]/[values] arrays of equal length, each
 * holding one nonzero entry. Immutable from the caller's perspective; to change the
 * sparsity pattern, rebuild.
 *
 * [get] is a linear scan rather than a binary search on purpose. Typical `nnz` is
 * small (handful to a few hundred for sparse feature vectors from nominal-heavy
 * CSPs), and at that scale a tight `IntArray` loop beats binary search's
 * mispredicted branches and indirect indexing. Internal ops iterate via
 * [forEachStored] and skip [get] entirely.
 *
 * Indices are not required to be sorted; the constructor only checks the parallel-
 * array invariant.
 *
 * @property size the logical length (including stored zeros).
 * @property indices the positions of the stored entries.
 * @property values the stored entry values, parallel to [indices].
 */
@Serializable
@SerialName("SparseVector")
class SparseVector internal constructor(override val size: Int, val indices: IntArray, val values: DoubleArray) :
    VectorView {

    init {
        require(indices.size == values.size) {
            "indices/values must align: ${indices.size} vs ${values.size}"
        }
    }

    override fun get(i: Int): Double {
        for (k in indices.indices) if (indices[k] == i) return values[k]
        return 0.0
    }

    override fun toDoubleArray(): DoubleArray {
        val out = DoubleArray(size)
        for (k in indices.indices) out[indices[k]] = values[k]
        return out
    }

    override fun equals(other: Any?): Boolean = this === other ||
        (
            other is SparseVector && size == other.size &&
                indices.contentEquals(other.indices) && values.contentEquals(other.values)
            )
    override fun hashCode(): Int {
        var h = size
        h = 31 * h + indices.contentHashCode()
        h = 31 * h + values.contentHashCode()
        return h
    }
    override fun toString(): String = "SparseVector(size=$size, nnz=${indices.size})"

    /** Factory entrypoints for [SparseVector]. */
    companion object {
        /** Build a sparse vector. Copies inputs so the caller can reuse the arrays. */
        fun of(size: Int, indices: IntArray, values: DoubleArray): SparseVector =
            SparseVector(size, indices.copyOf(), values.copyOf())
    }
}
