@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.Pinned
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.pin

/**
 * Operands held in place for the duration of one call.
 *
 * Kotlin/Native hands BLAS the caller's own storage. Every operand is pinned and passed as an interior
 * pointer, so there is no copy in either direction and no transfer to name in the route; that is the
 * substantive difference from the JVM binding, where a non-critical downcall cannot take heap memory at all.
 *
 * Every pin is released in a `finally`, so an exception thrown out of a call does not leave storage pinned.
 */
internal class Pins {
    private val pinned = ArrayList<Pinned<DoubleArray>>(PINS)

    /**
     * Pins [array] and returns a pointer to its entry at [index].
     *
     * An operand with no entries has no address to take: `addressOf` on an empty array raises rather than
     * returning a pointer nothing would dereference. BLAS still takes such an operand, because a zero depth
     * is a defined call that scales the destination, so a placeholder is pinned in its place. The extent
     * passed beside it is zero, so the library never reads through it.
     */
    fun pointer(array: DoubleArray, index: Int): CPointer<DoubleVar> {
        val storage = if (array.isEmpty()) EMPTY_OPERAND else array
        val pin = storage.pin()
        pinned += pin
        return pin.addressOf(if (storage === array) index else 0)
    }

    /** Releases every pin taken for this call. */
    fun release() {
        for (pin in pinned) pin.unpin()
        pinned.clear()
    }

    private companion object {
        const val PINS = 4

        /** Pinned in place of an operand with no entries, so a zero extent has an address to carry. */
        val EMPTY_OPERAND = DoubleArray(1)
    }
}

/** A vector operand reaching BLAS in place. */
internal class PinnedVector(
    /** Pointer to the lowest entry the vector reaches, which is where BLAS starts. */
    val pointer: CPointer<DoubleVar>,
    /** The BLAS increment, which keeps the vector's sign. */
    val increment: Int,
)

/** Pins [vector] for the call. Negative strides pass the low end with the sign kept on the increment. */
internal fun Pins.stage(vector: DenseVector): PinnedVector =
    PinnedVector(pointer(vector.data, baseIndex(vector)), vector.stride)

/** A matrix operand reaching BLAS in place, as the whole contiguous column-major block. */
internal class PinnedMatrix(
    /** Pointer BLAS reads and writes through. */
    val pointer: CPointer<DoubleVar>,
    /** The leading dimension, which for contiguous column-major storage is the row count. */
    val leadingDimension: Int,
)

/** Pins [matrix] for the call. */
internal fun Pins.stage(matrix: DenseMatrix): PinnedMatrix =
    PinnedMatrix(pointer(matrix.data, 0), maxOf(1, matrix.rows))

/**
 * The lowest storage index the vector touches, which is the pointer BLAS is handed.
 *
 * For a positive stride that is the vector's own offset. For a negative one it is the far end, because BLAS
 * walks a negatively stepped vector from the lowest address upward and treats the last element it reaches as
 * the logical first.
 */
internal fun baseIndex(vector: DenseVector): Int = baseIndex(vector.offset, vector.stride, vector.size)

/** The same rule over a raw run, which is what the Level 1 entry points hand BLAS. */
internal fun baseIndex(offset: Int, stride: Int, size: Int): Int =
    if (stride >= 0) offset else offset + (size - 1) * stride
