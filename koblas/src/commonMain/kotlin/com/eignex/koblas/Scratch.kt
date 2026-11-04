package com.eignex.koblas

import com.eignex.koblas.internal.operands.PooledBuffers

/**
 * Reusable temporary arrays a Level 2 or 3 routine borrows for alias snapshots, packed panels and sparse scratch.
 *
 * Passed to one call at a time, so it must not be shared by concurrent calls; nested calls may share it.
 * A call given none allocates its own temporary storage. Buffers never retain an operand and are returned even if
 * arithmetic throws.
 *
 * Reuse is by exact length, keeping repeated shapes allocation-free without exposing stale tail entries.
 * Retention keeps a bounded number of recently returned lengths, so changing shapes cannot grow the pool
 * indefinitely. Active loans remain valid through nested calls.
 */
public class Scratch {
    private val doubles = PooledBuffers<DoubleArray>()
    private val indices = PooledBuffers<IntArray>()

    internal fun take(size: Int): DoubleArray = doubles.take(size, ::DoubleArray)

    internal fun takeI32(size: Int): IntArray = indices.take(size, ::IntArray)

    internal fun release(buffer: DoubleArray): Unit = doubles.release(buffer)

    internal fun release(buffer: IntArray): Unit = indices.release(buffer)

    /** Idle floating-point buffers of [size]; an implementation diagnostic tests read to see reuse happen. */
    internal fun available(size: Int): Int = doubles.available(size)

    /** Idle index buffers of [size]; the counterpart of [available]. */
    internal fun availableI32(size: Int): Int = indices.available(size)

    /** Distinct idle floating-point lengths retained, which is what the retention bound is over. */
    internal fun idleLengths(): Int = doubles.idleLengths()

    /** Distinct idle index lengths retained; the counterpart of [idleLengths]. */
    internal fun idleI32Lengths(): Int = indices.idleLengths()
}
