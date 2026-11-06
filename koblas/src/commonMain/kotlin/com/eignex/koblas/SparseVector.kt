package com.eignex.koblas

import com.eignex.koblas.internal.storage.*
import com.eignex.koblas.internal.storage.sparse.lowerBound
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.jvm.JvmStatic

/**
 * Indices are strictly ascending and in range, validated by the constructor; [of] sorts and sums instead.
 *
 * Use [copyIndices] or [com.eignex.koblas.forEachStored] for safe structural access. [indices] is a live
 * zero-copy escape hatch
 * for specialized kernels and requires [UnsafeKoblasApi]; mutating it can invalidate the sparse structure.
 * [values] remains live so coefficients can be updated without changing the sparse pattern.
 * Inserting an entry through [set] or changing the pattern through [Vector.copyFrom] replaces the buffers;
 * obtain current arrays from the accessors after structural writes.
 *
 * @property size the logical length, counting the unstored zeros.
 */
@Serializable
@SerialName("SparseVector")
public class SparseVector internal constructor(
    override val size: Int,
    @SerialName("indices") private var storedIndices: IntArray,
    @SerialName("values") private var storedValues: DoubleArray,
) : VectorStorage {

    /** Live stored positions; structural writes may replace this buffer. Do not mutate it. */
    @property:UnsafeKoblasApi
    public val indices: IntArray get() = storedIndices

    /** Live stored values; structural writes may replace this buffer. */
    public val values: DoubleArray get() = storedValues

    init {
        requireShape(size >= 0) { "negative size: $size" }
        requireShape(indices.size == values.size) {
            "indices/values must align: ${indices.size} vs ${values.size}"
        }
        for (k in indices.indices) {
            requireIndex(indices[k] in 0 until size) { "indices[$k]=${indices[k]} out of [0,$size)" }
            require(k == 0 || indices[k - 1] < indices[k]) {
                "indices must be strictly ascending; found ${indices[k - 1]} then ${indices[k]} at $k"
            }
        }
    }

    /** Number of stored entries, counted as [SparseMatrix.nnz] counts a matrix's. */
    public val nnz: Int get() = values.size

    /** The stored value at (i), or 0.0 where nothing is stored. */
    override fun get(i: Int): Double {
        requireInBounds(i, size)
        val position = lowerBound(indices, 0, nnz, i)
        return if (position < nnz && indices[position] == i) values[position] else 0.0
    }

    /** Writes an entry, inserting its position when absent. Explicitly written zeros remain stored. */
    override fun set(i: Int, v: Double) {
        requireInBounds(i, size)
        val position = lowerBound(indices, 0, nnz, i)
        if (position < nnz && indices[position] == i) {
            values[position] = v
            return
        }
        requireShape(nnz < Int.MAX_VALUE) { "stored entry count exceeds Int capacity" }
        val slot = position
        val newIndices = IntArray(nnz + 1)
        val newValues = DoubleArray(nnz + 1)
        indices.copyInto(newIndices, 0, 0, slot)
        values.copyInto(newValues, 0, 0, slot)
        newIndices[slot] = i
        newValues[slot] = v
        indices.copyInto(newIndices, slot + 1, slot)
        values.copyInto(newValues, slot + 1, slot)
        replaceStorage(newIndices, newValues)
    }

    internal fun replaceStorage(indices: IntArray, values: DoubleArray) {
        storedIndices = indices
        storedValues = values
    }

    override fun toDoubleArray(): DoubleArray {
        val out = DoubleArray(size)
        for (k in indices.indices) out[indices[k]] = values[k]
        return out
    }

    /** A copy of the stored positions, in strictly ascending order. */
    public fun copyIndices(): IntArray = indices.copyOf()

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
    override fun toString(): String = "SparseVector(size=$size, nnz=$nnz)"

    /** Factories for sparse vectors. */
    public companion object {
        /** Build a sparse vector, sorting by index and summing duplicates. Copies its inputs. */
        @JvmStatic
        public fun of(size: Int, indices: IntArray, values: DoubleArray): SparseVector {
            requireShape(size >= 0) { "negative size: $size" }
            requireShape(indices.size == values.size) {
                "indices/values must align: ${indices.size} vs ${values.size}"
            }
            // Packing the index above the original position preserves input order when summing duplicates.
            val order = LongArray(indices.size) { (indices[it].toLong() shl Int.SIZE_BITS) or it.toLong() }
            order.sort()
            val idx = IntArray(indices.size)
            val vals = DoubleArray(values.size)
            var n = 0
            for (encoded in order) {
                val k = encoded.toInt()
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
         * Wrap existing arrays without copying, taking ownership; [indices] must already be strictly
         * ascending and in range. The structural indices cannot be recovered for mutation afterwards.
         */
        @JvmStatic
        public fun wrap(size: Int, indices: IntArray, values: DoubleArray): SparseVector = SparseVector(
            size,
            indices,
            values,
        )
    }
}
