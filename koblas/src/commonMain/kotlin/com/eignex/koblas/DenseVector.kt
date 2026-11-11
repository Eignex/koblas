package com.eignex.koblas

import com.eignex.koblas.internal.storage.*
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.jvm.JvmOverloads
import kotlin.jvm.JvmStatic

/**
 * Mutable dense vector, including borrowed slices with positive or negative strides.
 * Entry `x(i)` is at `offset + i * stride` in [values]. Factories allocate or borrow storage; [view]
 * shares it. Equality and hashing compare logical entries, including signed zeros and NaNs.
 * Serialization writes logical entries in order and decodes independent contiguous storage.
 * Do not use the vector as a hash-map key while mutating its entries.
 *
 * @property values the shared backing buffer, which may include entries outside this vector.
 * @property offset the first logical entry's buffer index, or an in-bounds boundary for an empty vector.
 * @property size the number of logical entries.
 * @property stride the nonzero physical distance between consecutive entries; negative strides are supported.
 */
@Serializable(with = DenseVectorSerializer::class)
@SerialName("DenseVector")
public class DenseVector internal constructor(
    public val values: DoubleArray,
    public val offset: Int,
    override val size: Int,
    public val stride: Int,
) : Vector {
    init {
        requireViewBounds(values.size, offset, size, stride, "vector")
    }

    override fun get(i: Int): Double {
        requireInBounds(i, size)
        return values[offset + i * stride]
    }

    override fun set(i: Int, v: Double) {
        requireInBounds(i, size)
        values[offset + i * stride] = v
    }

    override fun toDoubleArray(): DoubleArray = if (stride == 1) {
        values.copyOfRange(offset, offset + size)
    } else {
        DoubleArray(size) { values[offset + it * stride] }
    }

    /**
     * Borrows a slice addressed in this vector's logical indices. The offset and stride compose with the
     * existing layout, and both endpoints must remain within this vector. An empty slice may start at [size].
     * A composed stride outside the [Int] range is rejected.
     */
    @JvmOverloads
    public fun view(offset: Int, size: Int, stride: Int = 1): DenseVector {
        requireViewBounds(this.size, offset, size, stride, "logical vector")
        val physicalStride = this.stride.toLong() * stride
        requireShape(physicalStride in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) {
            "composed stride $physicalStride exceeds Int capacity"
        }
        val physicalOffset = this.offset.toLong() + offset.toLong() * this.stride
        // A stepped logical end may lie beyond either buffer end, but an empty slice addresses no entries.
        val origin = if (size == 0) physicalOffset.coerceIn(0L, values.size.toLong()) else physicalOffset
        return DenseVector(values, origin.toInt(), size, physicalStride.toInt())
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DenseVector || size != other.size) return false
        for (i in 0 until size) {
            val value = values[offset + i * stride]
            val otherValue = other.values[other.offset + i * other.stride]
            if (value.toBits() != otherValue.toBits()) return false
        }
        return true
    }

    override fun hashCode(): Int {
        var hash = 1
        for (i in 0 until size) hash = 31 * hash + values[offset + i * stride].hashCode()
        return hash
    }

    override fun toString(): String = "DenseVector(size=$size, offset=$offset, stride=$stride)"

    /** Factories for allocated or borrowed dense vector storage. */
    public companion object {
        /** Copies [values] into independent contiguous storage. */
        @JvmStatic
        public fun of(values: DoubleArray): DenseVector = DenseVector(values.copyOf(), 0, values.size, 1)

        /** Allocates a zero-filled contiguous vector of [size] entries. */
        @JvmStatic
        public fun zero(size: Int): DenseVector {
            requireShape(size >= 0) { "negative size: $size" }
            return DenseVector(DoubleArray(size), 0, size, 1)
        }

        /**
         * Borrows [size] entries of [values], starting at [offset] with the given [stride]. No entries are
         * copied; mutations through either reference are shared. The default size covers the remaining
         * buffer at unit stride; specify the logical size for other strides. Negative strides are supported.
         * Empty vectors require an offset in `0..values.size`. Bounds use overflow-safe arithmetic.
         */
        @JvmStatic
        @JvmOverloads
        public fun wrap(
            values: DoubleArray,
            offset: Int = 0,
            size: Int = values.size - offset,
            stride: Int = 1,
        ): DenseVector = DenseVector(values, offset, size, stride)
    }
}

/** An empty view addresses no entry, yet its origin must still lie within the buffer or at its end. */
private fun requireViewBounds(bufferSize: Int, offset: Int, size: Int, stride: Int, description: String) {
    requireVectorWindow(bufferSize, offset, size, stride, "$description view")
    requireShape(size != 0 || offset in 0..bufferSize) {
        "$description view offset $offset exceeds buffer length $bufferSize"
    }
}
