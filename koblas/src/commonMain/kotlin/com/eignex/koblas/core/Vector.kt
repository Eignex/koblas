package com.eignex.koblas.core

import com.eignex.koblas.*
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** Read-only vector contract. Anything that only reads a vector should take this. */
public interface F64VectorLike {
    /** Number of entries, counting the unstored zeros of a sparse vector. */
    public val size: Int

    /** The entry at index (i). Throws `IndexOutOfBoundsException` outside `0 until size`, whatever the storage. */
    public operator fun get(i: Int): Double

    /** Materialise into a fresh dense `DoubleArray`, independent of the internal storage. */
    public fun toDoubleArray(): DoubleArray
}

/** The vector storages koblas itself defines, [F64DenseVector] and [F64SparseVector]. */
@Serializable
public sealed interface F64VectorView : F64VectorLike

/**
 * @property data the flat backing array. The vector is mutable through it and [set]; do not use the vector as
 *   a hash-map key while mutating it.
 */
@Serializable
@SerialName("F64DenseVector")
public class F64DenseVector internal constructor(public val data: DoubleArray) : F64VectorView {

    /** This storage owns [data], including when ownership was transferred through [wrap]. */
    public val ownership: BufferOwnership get() = BufferOwnership.OWNED

    internal constructor(size: Int) : this(DoubleArray(size))

    override val size: Int get() = data.size

    override fun get(i: Int): Double {
        requireInBounds(i, size)
        return data[i]
    }

    override fun toDoubleArray(): DoubleArray = data.copyOf()

    /** Writes (v) at index (i). */
    public operator fun set(i: Int, v: Double) {
        requireInBounds(i, size)
        data[i] = v
    }

    override fun equals(other: Any?): Boolean =
        this === other || (other is F64DenseVector && data.contentEquals(other.data))
    override fun hashCode(): Int = data.contentHashCode()
    override fun toString(): String = "F64DenseVector(size=$size)"

    /** Factories for dense vectors. */
    public companion object {
        /** Copy a `DoubleArray` into a fresh dense vector. */
        public fun of(values: DoubleArray): F64DenseVector = F64DenseVector(values.copyOf())

        /** A dense vector of [size] zeros. */
        public fun zero(size: Int): F64DenseVector {
            requireShape(size >= 0) { "negative size: $size" }
            return F64DenseVector(size)
        }

        /** Wrap an existing `DoubleArray` without copying. The caller relinquishes ownership. */
        public fun wrap(data: DoubleArray): F64DenseVector = F64DenseVector(data)
    }
}

/**
 * Indices are strictly ascending and in range, validated by the constructor; [of] sorts and sums instead.
 *
 * Use [copyIndices] or [com.eignex.koblas.forEachStored] for safe structural access. [indices] is a live
 * zero-copy escape hatch
 * for specialized kernels and requires [UnsafeKoblasApi]; mutating it can invalidate the sparse structure.
 * [values] remains live so coefficients can be updated without changing the sparse pattern.
 *
 * @property size the logical length, counting the unstored zeros.
 * @property indices live positions of the stored entries; do not mutate.
 * @property values the stored entry values, parallel to the stored positions.
 */
@Serializable
@SerialName("F64SparseVector")
public class F64SparseVector internal constructor(
    override val size: Int,
    @property:UnsafeKoblasApi public val indices: IntArray,
    public val values: DoubleArray,
) : F64VectorView {

    init {
        requireShape(size >= 0) { "negative size: $size" }
        requireShape(indices.size == values.size) {
            "indices/values must align: ${indices.size} vs ${values.size}"
        }
        for (k in indices.indices) {
            requireShape(indices[k] in 0 until size) { "indices[$k]=${indices[k]} out of [0,$size)" }
            require(k == 0 || indices[k - 1] < indices[k]) {
                "indices must be strictly ascending; found ${indices[k - 1]} then ${indices[k]} at $k"
            }
        }
    }

    /** The stored value at (i), or 0.0 where nothing is stored. */
    override fun get(i: Int): Double {
        requireInBounds(i, size)
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

    /** A copy of the stored positions, in strictly ascending order. */
    public fun copyIndices(): IntArray = indices.copyOf()

    override fun equals(other: Any?): Boolean = this === other ||
        (
            other is F64SparseVector && size == other.size &&
                indices.contentEquals(other.indices) && values.contentEquals(other.values)
            )
    override fun hashCode(): Int {
        var h = size
        h = 31 * h + indices.contentHashCode()
        h = 31 * h + values.contentHashCode()
        return h
    }
    override fun toString(): String = "F64SparseVector(size=$size, nnz=${indices.size})"

    /** Factories for sparse vectors. */
    public companion object {
        /** Build a sparse vector, sorting by index and summing duplicates. Copies its inputs. */
        public fun of(size: Int, indices: IntArray, values: DoubleArray): F64SparseVector {
            requireShape(indices.size == values.size) {
                "indices/values must align: ${indices.size} vs ${values.size}"
            }
            // Index in the high half and position in the low, so one primitive sort orders by index and
            // keeps equal indices in the order given, which is what summing duplicates in one pass needs.
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
            return F64SparseVector(size, idx.copyOf(n), vals.copyOf(n))
        }

        /**
         * Wrap existing arrays without copying, taking ownership; [indices] must already be strictly
         * ascending and in range. The structural indices cannot be recovered for mutation afterwards.
         */
        public fun wrap(size: Int, indices: IntArray, values: DoubleArray): F64SparseVector = F64SparseVector(
            size,
            indices,
            values,
        )
    }
}
