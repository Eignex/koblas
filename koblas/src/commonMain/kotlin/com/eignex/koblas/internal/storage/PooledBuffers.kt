package com.eignex.koblas.internal.storage

import com.eignex.koblas.*

/**
 * Primitive buffers lent by exact length and retained under a bound on idle lengths.
 *
 * [MAX_IDLE_LENGTHS] accommodates repeated shapes; evicting the least recently returned length bounds
 * retention across changing shapes. Only lent buffers can be returned, bounding the count within a length.
 *
 * Linear searches avoid a map allocation per length. The idle list is bounded, and the widest schedule
 * holds seven loans at once, so both lists remain small.
 */
internal class PooledBuffers<A : Any> {
    private val idle = ArrayList<A>()
    private val lent = ArrayList<A>()

    fun take(size: Int, allocate: (Int) -> A): A {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        val index = idleWithSize(size)
        val buffer = if (index >= 0) idle.removeAt(index) else allocate(size)
        lent += buffer
        return buffer
    }

    fun release(buffer: A) {
        val index = lentIndexOf(buffer)
        check(index >= 0) { "released a buffer this scratch did not lend" }
        lent.removeAt(index)
        makeRoomFor(sizeOf(buffer))
        idle += buffer
    }

    fun available(size: Int): Int {
        var count = 0
        for (i in idle.indices) if (sizeOf(idle[i]) == size) count++
        return count
    }

    /** Counts distinct idle lengths without allocating a set on every release. */
    fun idleLengths(): Int {
        var count = 0
        for (i in idle.indices) {
            var seen = false
            for (j in 0 until i) {
                if (sizeOf(idle[j]) == sizeOf(idle[i])) {
                    seen = true
                    break
                }
            }
            if (!seen) count++
        }
        return count
    }

    /** Evicts all buffers of the least recently returned length when [size] would exceed the bound. */
    private fun makeRoomFor(size: Int) {
        if (idleWithSize(size) >= 0 || idleLengths() < MAX_IDLE_LENGTHS) return
        val oldest = sizeOf(idle[0])
        for (i in idle.lastIndex downTo 0) if (sizeOf(idle[i]) == oldest) idle.removeAt(i)
    }

    private fun idleWithSize(size: Int): Int {
        for (i in idle.indices) if (sizeOf(idle[i]) == size) return i
        return -1
    }

    private fun lentIndexOf(buffer: A): Int {
        for (i in lent.indices) if (lent[i] === buffer) return i
        return -1
    }

    private fun sizeOf(buffer: A): Int = when (buffer) {
        is DoubleArray -> buffer.size
        is IntArray -> buffer.size
        else -> error("unsupported workspace buffer")
    }
}

/**
 * Distinct idle lengths retained per primitive element type.
 *
 * Eight covers the lengths one sparse Level 3 call asks for several times over: a right-hand-side panel, a
 * diagonal, a staged operand, an accumulator and the rank-update index scratch, with room for a caller
 * alternating between two shapes. A ninth length evicts the least recently returned one.
 */
internal const val MAX_IDLE_LENGTHS = 8

internal inline fun <T> Scratch?.borrowI32(size: Int, block: (IntArray) -> T): T {
    val buffer = this?.takeI32(size) ?: IntArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

private val EMPTY_BUFFER = DoubleArray(0)

/** Unused panels consume neither a loan nor a retained length. */
internal inline fun <T> Scratch?.borrowOptional(size: Int, block: (DoubleArray) -> T): T =
    if (size == 0) block(EMPTY_BUFFER) else borrow(size, block)
