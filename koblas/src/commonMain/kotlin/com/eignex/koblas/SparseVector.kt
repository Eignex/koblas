package com.eignex.koblas

import com.eignex.koblas.internal.storage.*
import com.eignex.koblas.internal.storage.sparse.lowerBound
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.jvm.JvmStatic

/**
 * Stored indices are strictly ascending and in range within this vector's window. Factories validate
 * the structure; [of] sorts and sums instead.
 *
 * Use [copyIndices] or [com.eignex.koblas.forEachStored] for safe structural access. [indices] is a live
 * zero-copy escape hatch
 * for specialized kernels and requires [UnsafeKoblasApi]; mutating it can invalidate the sparse structure.
 * [values] remains live so coefficients can be updated without changing the sparse pattern. Column views
 * borrow their owner's arrays: only the [nnz] entries beginning at [offset] belong to the vector. A view
 * follows its matrix after structural changes; serialization and copies contain only its stored entries.
 * Inserting an entry through [set] or changing the pattern through [Vector.copyFrom] or [KoblasEngine.axpy]
 * replaces the buffers;
 * obtain current arrays from the accessors after structural writes.
 *
 * @property size the logical length, counting the unstored zeros.
 */
@Serializable(with = SparseVectorSerializer::class)
@SerialName("SparseVector")
public class SparseVector internal constructor(
    override val size: Int,
    @SerialName("indices") private var storedIndices: IntArray,
    @SerialName("values") private var storedValues: DoubleArray,
    private val columnOwner: SparseMatrix? = null,
    private val columnIndex: Int = 0,
) : VectorStorage {

    /** Live stored positions; structural writes may replace this buffer. Do not mutate it. */
    @property:UnsafeKoblasApi
    public val indices: IntArray get() = columnOwner?.rowIndices ?: storedIndices

    /** Live stored values; structural writes may replace this buffer. */
    public val values: DoubleArray get() = columnOwner?.values ?: storedValues

    /** Start of this vector's stored entries in [indices] and [values]. Owning vectors start at zero. */
    public val offset: Int get() = columnOwner?.colPointers?.get(columnIndex) ?: 0

    init {
        requireShape(size >= 0) { "negative size: $size" }
        if (columnOwner == null) {
            requireShape(indices.size == values.size) { "indices/values must align: ${indices.size} vs ${values.size}" }
            for (k in indices.indices) {
                requireIndex(indices[k] in 0 until size) { "indices[$k]=${indices[k]} out of [0,$size)" }
                require(k == 0 || indices[k - 1] < indices[k]) {
                    "indices must be strictly ascending; found ${indices[k - 1]} then ${indices[k]} at $k"
                }
            }
        } else {
            requireInBounds(columnIndex, columnOwner.cols)
        }
    }

    /** Number of stored entries, beginning at [offset] in the backing arrays. */
    public val nnz: Int get() = columnOwner?.let { it.colPointers[columnIndex + 1] - offset } ?: storedValues.size

    /** The stored value at (i), or 0.0 where nothing is stored. */
    override fun get(i: Int): Double {
        requireInBounds(i, size)
        val position = lowerBound(indices, offset, offset + nnz, i)
        return if (position < offset + nnz && indices[position] == i) values[position] else 0.0
    }

    /** Writes an entry, inserting its position when absent. Explicitly written zeros remain stored. */
    override fun set(i: Int, v: Double) {
        requireInBounds(i, size)
        val owner = columnOwner
        if (owner != null) {
            owner[i, columnIndex] = v
            return
        }
        val position = lowerBound(indices, offset, offset + nnz, i)
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
        val start = offset
        val currentIndices = this.indices
        if (indices.size == nnz && indices.indices.all { indices[it] == currentIndices[start + it] }) {
            values.copyInto(this.values, start)
        } else {
            val owner = columnOwner
            if (owner == null) {
                storedIndices = indices
                storedValues = values
            } else {
                owner.replaceColumn(columnIndex, indices, values)
            }
        }
    }

    override fun toDoubleArray(): DoubleArray {
        val out = DoubleArray(size)
        for (k in offset until offset + nnz) out[indices[k]] = values[k]
        return out
    }

    /** A copy of this vector's stored positions, in strictly ascending order. */
    public fun copyIndices(): IntArray = indices.copyOfRange(offset, offset + nnz)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SparseVector || size != other.size || nnz != other.nnz) return false
        val start = offset
        val otherStart = other.offset
        for (k in 0 until nnz) {
            if (indices[start + k] != other.indices[otherStart + k] ||
                values[start + k].toBits() != other.values[otherStart + k].toBits()
            ) return false
        }
        return true
    }

    override fun hashCode(): Int {
        var indexHash = 1
        var valueHash = 1
        for (k in offset until offset + nnz) {
            indexHash = 31 * indexHash + indices[k]
            valueHash = 31 * valueHash + values[k].hashCode()
        }
        return (size * 31 + indexHash) * 31 + valueHash
    }
    override fun toString(): String = "SparseVector(size=$size, nnz=$nnz)"

    /** Factories for sparse vectors. */
    public companion object {
        private val emptyIndices = IntArray(0)
        private val emptyValues = DoubleArray(0)

        internal fun columnView(owner: SparseMatrix, column: Int): SparseVector =
            SparseVector(owner.rows, emptyIndices, emptyValues, owner, column)

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
