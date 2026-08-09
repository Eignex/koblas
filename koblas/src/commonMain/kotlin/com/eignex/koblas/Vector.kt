package com.eignex.koblas

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Read-only vector contract: length, entry access, materialisation. Anything that only reads a vector should
 * take this.
 *
 * Open, for the reasons [MatrixLike] gives: a caller may already have a shape koblas does not define, and
 * copying into a [DenseVector] to use koblas's routines was a tax on the callers with the most data. koblas
 * recognises its own storages by type and sweeps them directly; a foreign implementation takes the generic
 * path through [get].
 *
 * @see VectorView the closed, serializable subset.
 */
interface VectorLike {
    /** Number of entries (including stored zeros for sparse). */
    val size: Int

    /**
     * Read entry at [i]. O(1) for [DenseVector], `O(log nnz)` for [SparseVector], which binary-searches
     * its ascending indices. Use the [forEachStored] extension when you want to walk the populated
     * entries without per-index lookup cost.
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
 * The vector storages koblas itself defines: [DenseVector] and [SparseVector], and nothing else ever.
 *
 * Sealed for serialization, exactly as [MatrixView] is — a snapshot decodes back into the storage it was
 * written from without a consumer registering anything. Take [VectorLike] instead unless you need the
 * closed set.
 *
 * The split between the two storages is a backing-storage choice, not a semantic one: dense pays per
 * coordinate, sparse pays per nonzero. Most code iterates via [forEachStored] to walk only the populated
 * entries, so sparse callers feed sparse vectors without the cost of dense materialisation and dense
 * callers walk every index the same way.
 */
@Serializable
sealed interface VectorView : VectorLike

/**
 * Dense double-precision vector backed by a flat `DoubleArray`. The
 * default carrier when the caller already has a dense array or expects
 * most entries to be populated.
 *
 * Construction goes through the [Companion] factories, as it does for every
 * koblas container: [of] copies a `DoubleArray`, [zero] allocates one of a
 * given size, and [wrap] adopts an existing array without copying. Unlike the
 * read-only [VectorView] contract, the concrete vector exposes its [data]
 * backing and elementwise [set] for in-place updates.
 *
 * @property data the flat backing array.
 */
@Serializable
@SerialName("DenseVector")
class DenseVector internal constructor(val data: DoubleArray) : VectorView {

    internal constructor(size: Int) : this(DoubleArray(size))

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
 * Indices are strictly ascending and in range, validated by the constructor. Three things depend on it:
 * [get] binary-searches rather than scanning, a sparse-against-sparse `dot` merges the two index lists in
 * one pass instead of looking each position up, and the storage order that `forEachStored` and `iamax`
 * expose becomes index order — so a tie in `iamax` resolves to the lowest index, the same rule the dense
 * vector follows. Strict ascent also rules out duplicate indices, which would otherwise leave [get] and
 * `forEachStored` disagreeing about the value at a position.
 *
 * [of] is the forgiving entry point: it sorts and sums duplicates, mirroring `SparseMatrix.ofColumns`.
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
        for (k in indices.indices) {
            require(indices[k] in 0 until size) { "indices[$k]=${indices[k]} out of [0,$size)" }
            require(k == 0 || indices[k - 1] < indices[k]) {
                "indices must be strictly ascending; found ${indices[k - 1]} then ${indices[k]} at $k"
            }
        }
    }

    /** The stored value at [i], or `0.0`. A binary search over the ascending indices, so `O(log nnz)`. */
    override fun get(i: Int): Double {
        var lo = 0
        var hi = indices.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val idx = indices[mid]
            when {
                idx < i -> lo = mid + 1
                idx > i -> hi = mid - 1
                else -> return values[mid]
            }
        }
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
        /**
         * Build a sparse vector, sorting by index and summing duplicates. Copies its inputs, so the
         * caller can reuse the arrays.
         *
         * The forgiving counterpart to the constructor, which requires the ordering it relies on. Entries
         * may arrive in any order; a repeated index contributes the sum of its values, which is what a
         * scatter of accumulated contributions means.
         */
        fun of(size: Int, indices: IntArray, values: DoubleArray): SparseVector {
            require(indices.size == values.size) {
                "indices/values must align: ${indices.size} vs ${values.size}"
            }
            val order = indices.indices.sortedBy { indices[it] }
            val idx = IntArray(indices.size)
            val vals = DoubleArray(values.size)
            var n = 0
            for (k in order) {
                if (n > 0 && idx[n - 1] == indices[k]) {
                    vals[n - 1] += values[k]
                } else {
                    idx[n] = indices[k]
                    vals[n] = values[k]
                    n++
                }
            }
            return SparseVector(size, idx.copyOf(n), vals.copyOf(n))
        }

        /**
         * Wrap existing index/value arrays without copying. Caller relinquishes ownership.
         *
         * The strict entry point, and the counterpart of [DenseVector.wrap]: [indices] must already be
         * strictly ascending and in range, which is validated rather than repaired. Use [of] when the
         * entries need sorting or duplicates need summing.
         */
        fun wrap(size: Int, indices: IntArray, values: DoubleArray): SparseVector = SparseVector(size, indices, values)
    }
}
