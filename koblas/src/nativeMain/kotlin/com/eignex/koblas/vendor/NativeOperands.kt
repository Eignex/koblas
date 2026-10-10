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
 * Operands pinned in place for one call. Interior pointers reach the caller's storage without
 * copies. [withPins] releases every pin in `finally`, including when a call throws.
 */
internal class Pins {
    private val pinned = ArrayList<Pinned<DoubleArray>>(PINS)

    /**
     * Pin [array] and return its entry at [index]. Empty operands use a placeholder because
     * `addressOf` requires storage, while BLAS allows zero extents, including a zero-depth product
     * that scales the destination. The library does not read the placeholder.
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
    PinnedVector(pointer(vector.values, baseIndex(vector)), vector.stride)

/** A matrix operand reaching BLAS in place, as the whole contiguous column-major block. */
internal class PinnedMatrix(
    /** Pointer BLAS reads and writes through. */
    val pointer: CPointer<DoubleVar>,
    /** The leading dimension, which for contiguous column-major storage is the row count. */
    val leadingDimension: Int,
)

/** Pins [matrix] for the call. */
internal fun Pins.stage(matrix: DenseMatrix): PinnedMatrix =
    PinnedMatrix(pointer(matrix.values, 0), maxOf(1, matrix.rows))

/**
 * Lowest storage index passed to BLAS. Negative increments retain their sign but start at the
 * low address, allowing BLAS to reconstruct the logical first entry at the high end.
 */
internal fun baseIndex(vector: DenseVector): Int = baseIndex(vector.offset, vector.stride, vector.size)

/** The same rule over a raw run, which is what the Level 1 entry points hand BLAS. */
internal fun baseIndex(offset: Int, stride: Int, size: Int): Int =
    if (stride >= 0) offset else offset + (size - 1) * stride

/** Releases every pin when [block] returns or throws. */
internal inline fun <T> withPins(block: (Pins) -> T): T {
    val pins = Pins()
    try {
        return block(pins)
    } finally {
        pins.release()
    }
}
