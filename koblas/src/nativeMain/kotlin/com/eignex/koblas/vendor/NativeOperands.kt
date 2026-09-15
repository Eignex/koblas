@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.Pinned
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.pin

/**
 * Operands held in place for the duration of one call.
 *
 * Kotlin/Native hands BLAS the caller's own storage. A directly addressed window is pinned and passed as an
 * interior pointer, so there is no copy in either direction and no transfer to name in the route; that is the
 * substantive difference from the JVM binding, where a non-critical downcall cannot take heap memory at all.
 *
 * Only a window BLAS cannot address is copied, and it is copied back after the call if the operation writes to
 * it. Every pin is released in a `finally`, so an exception thrown out of a call does not leave storage pinned.
 */
internal class Pins {
    private val pinned = ArrayList<Pinned<DoubleArray>>(PINS)

    /** Pins [array] and returns a pointer to its entry at [index]. */
    fun pointer(array: DoubleArray, index: Int): CPointer<DoubleVar> {
        val pin = array.pin()
        pinned += pin
        return pin.addressOf(index)
    }

    /** Releases every pin taken for this call. */
    fun release() {
        for (pin in pinned) pin.unpin()
        pinned.clear()
    }

    private companion object {
        const val PINS = 4
    }
}

/** A vector operand reaching BLAS in place. */
internal class PinnedVector(
    /** Pointer to the lowest entry the window reaches, which is where BLAS starts. */
    val pointer: CPointer<DoubleVar>,
    /** The BLAS increment, which keeps the window's sign. */
    val increment: Int,
)

/** Pins [window] for the call. Negative strides pass the low end with the sign kept on the increment. */
internal fun Pins.stage(window: VectorWindow): PinnedVector =
    PinnedVector(pointer(window.data, baseIndex(window)), window.stride)

/** A matrix operand, in place when BLAS can address it and packed when it cannot. */
internal class PinnedMatrix(
    /** Pointer BLAS reads and writes through. */
    val pointer: CPointer<DoubleVar>,
    /** The leading dimension that goes with [addressing]. */
    val leadingDimension: Int,
    /** How the window reached BLAS, which decides whether anything has to come back. */
    val addressing: Addressing,
    private val window: MatrixWindow,
    private val packed: DoubleArray?,
) {
    /** Unpacks a staged block back over its window. A directly addressed one was written in place already. */
    fun writeBack() {
        if (packed != null) unstageFrom(packed, window)
    }
}

/** Pins or packs [window] for the call under [addressing]. */
internal fun Pins.stage(window: MatrixWindow, addressing: Addressing): PinnedMatrix {
    if (addressing == Addressing.Staged) {
        val packed = DoubleArray(maxOf(1, window.rows * window.columns))
        stageInto(window, packed)
        return PinnedMatrix(pointer(packed, 0), maxOf(1, window.rows), addressing, window, packed)
    }
    return PinnedMatrix(
        pointer(window.data, window.offset),
        leadingDimension(window, addressing),
        addressing,
        window,
        null,
    )
}
