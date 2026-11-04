@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.operands.requireVectorWindow
import com.eignex.koblas.internal.operands.windowsOverlap
import kotlin.jvm.JvmOverloads

/**
 * A mutable live window of [size] entries in [values], starting at [offset] and separated by [stride].
 *
 * The stepped shape of [DenseVector]: every entry is stored, so a vendor addresses it as a pointer and an
 * increment exactly as it does a contiguous one. Negative stride is supported when both ends remain in the
 * buffer. This shape always borrows [values] rather than copying it, so mutations through it or through any
 * other reference to the array are visible to each other.
 */
public class StridedVector @JvmOverloads constructor(
    public override val values: DoubleArray,
    public override val offset: Int,
    override val size: Int,
    public override val stride: Int = 1,
) : DenseVector {
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

    override fun toDoubleArray(): DoubleArray = DoubleArray(size) { get(it) }

    override fun toString(): String = "StridedVector(size=$size, offset=$offset, stride=$stride)"
}

/** Whether these vectors address at least one common buffer entry. */
public fun DenseVector.overlaps(other: DenseVector): Boolean =
    windowsOverlap(values, offset, size, stride, other.values, other.offset, other.size, other.stride)

/** A borrowed view over this whole vector, keeping whatever origin and spacing it already has. */
public fun DenseVector.asView(): StridedVector = StridedVector(values, offset, size, stride)

/**
 * A borrowed strided slice, addressed in this vector's logical indices rather than its buffer's positions.
 *
 * The receiver's own origin and spacing compose with the ones asked for, so a slice of a slice addresses what
 * the outer one addressed. Reading the arguments as buffer positions instead would silently move the window
 * whenever the receiver was itself a view.
 *
 * Both endpoints must lie within this vector's logical bounds. An empty slice may start at [DenseVector.size].
 * A composed stride that cannot be represented as an [Int] is rejected.
 */
@JvmOverloads
public fun DenseVector.view(offset: Int, size: Int, stride: Int = 1): StridedVector {
    requireViewBounds(this.size, offset, size, stride, "logical vector")
    val physicalStride = this.stride.toLong() * stride
    requireShape(physicalStride in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) {
        "composed stride $physicalStride exceeds Int capacity"
    }
    val physicalOffset = this.offset.toLong() + offset.toLong() * this.stride
    // An empty slice has no physical entries; a stepped logical end can lie beyond either buffer end.
    val origin = if (size == 0) physicalOffset.coerceIn(0L, values.size.toLong()) else physicalOffset
    return StridedVector(values, origin.toInt(), size, physicalStride.toInt())
}

/** An empty view addresses no entry, yet its origin must still lie within the buffer or at its end. */
private fun requireViewBounds(bufferSize: Int, offset: Int, size: Int, stride: Int, description: String) {
    requireVectorWindow(bufferSize, offset, size, stride, "$description view")
    requireShape(size != 0 || offset in 0..bufferSize) {
        "$description view offset $offset exceeds buffer length $bufferSize"
    }
}
