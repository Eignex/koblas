@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.requireInBounds
import com.eignex.koblas.requireNonNegativeShape
import com.eignex.koblas.requireShape
import kotlin.jvm.JvmOverloads

/**
 * A mutable live window of [size] entries in [data], starting at [offset] and separated by [stride].
 *
 * The stepped shape of [DenseVector]: every entry is stored, so a vendor addresses it as a pointer and an
 * increment exactly as it does a contiguous one. Negative stride is supported when both ends remain in the
 * buffer. This shape always borrows [data] rather than copying it, so mutations through it or through any
 * other reference to the array are visible to each other.
 */
public class StridedVector @JvmOverloads constructor(
    public override val data: DoubleArray,
    public override val offset: Int,
    override val size: Int,
    public override val stride: Int = 1,
) : DenseVector {
    init {
        requireShape(size >= 0) { "negative size: $size" }
        require(stride != 0) { "stride must not be zero" }
        requireViewBounds(data.size, offset, size, stride, "vector")
    }

    override fun get(i: Int): Double {
        requireInBounds(i, size)
        return data[offset + i * stride]
    }

    override fun set(i: Int, value: Double) {
        requireInBounds(i, size)
        data[offset + i * stride] = value
    }

    override fun toDoubleArray(): DoubleArray = DoubleArray(size) { get(it) }

    override fun toString(): String = "StridedVector(size=$size, offset=$offset, stride=$stride)"
}

/** Whether these vectors address at least one common buffer entry. */
public fun StridedVector.overlaps(other: StridedVector): Boolean {
    if (data !== other.data) return false
    val first = if (size <= other.size) this else other
    val second = if (first === this) other else this
    for (i in 0 until first.size) {
        val physical = first.offset + i * first.stride
        val relative = physical - second.offset
        if (relative % second.stride == 0 && relative / second.stride in 0 until second.size) return true
    }
    return false
}

/** A borrowed view over this entire owned vector. */
public fun DenseVector.asView(): StridedVector = StridedVector(data, 0, size)

/** A borrowed strided slice of this owned vector. */
@JvmOverloads
public fun DenseVector.view(offset: Int, size: Int, stride: Int = 1): StridedVector =
    StridedVector(data, offset, size, stride)

private fun requireViewBounds(bufferSize: Int, offset: Int, size: Int, stride: Int, description: String) {
    if (size == 0) {
        requireShape(offset in 0..bufferSize) { "$description view offset $offset exceeds buffer length $bufferSize" }
        return
    }
    val last = offset.toLong() + (size - 1).toLong() * stride
    val firstPhysical = minOf(offset.toLong(), last)
    val lastPhysical = maxOf(offset.toLong(), last)
    requireShape(firstPhysical >= 0 && lastPhysical < bufferSize) {
        "$description view addresses [$firstPhysical, $lastPhysical] outside buffer length $bufferSize"
    }
}
