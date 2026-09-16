package com.eignex.koblas

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.jvm.JvmStatic

/** Read-only vector contract. Anything that only reads a vector should take this. */
public interface Vector {
    /** Number of entries, counting the unstored zeros of a sparse vector. */
    public val size: Int

    /** The entry at index (i). Throws `IndexOutOfBoundsException` outside `0 until size`, whatever the storage. */
    public operator fun get(i: Int): Double

    /** Materialise into a fresh dense `DoubleArray`, independent of the internal storage. */
    public fun toDoubleArray(): DoubleArray
}

/**
 * The vector storages koblas defines and serializes, [ContiguousVector] and [SparseVector].
 *
 * This is about which types koblas owns the definition of, not about which own their buffers: either may be
 * wrapped around an array the caller keeps a reference to.
 */
@Serializable
public sealed interface VectorStorage : Vector

/**
 * A vector that stores every one of its entries, addressed as a buffer, an origin and a step.
 *
 * Dense is about what is stored, not about how it is spaced: a run of adjacent entries and every second entry
 * of a longer buffer are both dense, and a BLAS call takes either as a pointer and an increment without
 * knowing the difference. That is why this is the operand type the dense matrix routines declare. A vector
 * storing a pattern is a different thing with no increment to hand over, so [SparseVector] is deliberately not
 * one of these and does not compile where one is wanted.
 *
 * Sealed because the two shapes are the whole set: entries laid out one after another, or entries a fixed
 * step apart. A caller that must tell them apart can do so exhaustively.
 *
 * Neither shape implies ownership. [of] and [zero] allocate, while [wrap] and the [StridedVector] constructor
 * borrow an array the caller keeps; mutations stay visible through every reference either way. So a vector
 * here says how its entries are spaced, and says nothing about who owns them.
 */
public sealed interface DenseVector : Vector {
    /** Borrowed or owned backing array. */
    public val data: DoubleArray

    /** First physical entry of this vector within [data]. */
    public val offset: Int

    /** Physical distance between adjacent entries, which may be negative but is never zero. */
    public val stride: Int

    /** Writes (v) at index (i). */
    public operator fun set(i: Int, v: Double)

    /**
     * Factories for dense vectors.
     *
     * Each names [ContiguousVector] rather than this interface, because every one of them produces that
     * shape and a caller that wants the serializable storage should not have to narrow the result back.
     */
    public companion object {
        /** Copy a `DoubleArray` into a fresh dense vector. */
        @JvmStatic
        public fun of(values: DoubleArray): ContiguousVector = ContiguousVector(values.copyOf())

        /** A dense vector of [size] zeros. */
        @JvmStatic
        public fun zero(size: Int): ContiguousVector {
            requireShape(size >= 0) { "negative size: $size" }
            return ContiguousVector(DoubleArray(size))
        }

        /** Wrap an existing `DoubleArray` without copying; mutations remain visible through both references. */
        @JvmStatic
        public fun wrap(data: DoubleArray): ContiguousVector = ContiguousVector(data)
    }
}

/**
 * A dense vector that owns its whole buffer, so its entries are `data` itself.
 *
 * @property data the flat backing array. The vector is mutable through it and [set]; do not use the vector as
 *   a hash-map key while mutating it.
 */
@Serializable
@SerialName("DenseVector")
public class ContiguousVector internal constructor(public override val data: DoubleArray) :
    VectorStorage,
    DenseVector {
    internal constructor(size: Int) : this(DoubleArray(size))

    override val size: Int get() = data.size
    override val offset: Int get() = 0
    override val stride: Int get() = 1

    override fun get(i: Int): Double {
        requireInBounds(i, size)
        return data[i]
    }

    override fun toDoubleArray(): DoubleArray = data.copyOf()

    override fun set(i: Int, v: Double) {
        requireInBounds(i, size)
        data[i] = v
    }

    override fun equals(other: Any?): Boolean =
        this === other || (other is ContiguousVector && data.contentEquals(other.data))
    override fun hashCode(): Int = data.contentHashCode()
    override fun toString(): String = "DenseVector(size=$size)"
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
@SerialName("SparseVector")
public class SparseVector internal constructor(
    override val size: Int,
    @property:UnsafeKoblasApi public val indices: IntArray,
    public val values: DoubleArray,
) : VectorStorage {

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

    /** Factories for sparse vectors. */
    public companion object {
        /** Build a sparse vector, sorting by index and summing duplicates. Copies its inputs. */
        @JvmStatic
        public fun of(size: Int, indices: IntArray, values: DoubleArray): SparseVector {
            requireShape(size >= 0) { "negative size: $size" }
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
