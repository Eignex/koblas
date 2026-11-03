package com.eignex.koblas.workspace

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.engine.*

/**
 * Reusable scratch for alias-safe matrix operations.
 *
 * A workspace belongs to one invocation at a time. Independent calls may use distinct workspaces; calls
 * without one own their temporary storage. Buffers never retain an operand and are returned even if
 * arithmetic throws.
 *
 * Reuse is by exact length, keeping repeated shapes allocation-free without exposing stale tail entries.
 * Retention keeps a bounded number of recently returned lengths, so changing shapes cannot grow the pool
 * indefinitely. Active loans remain valid through nested calls.
 */
public class Workspace {
    private val doubles = PooledBuffers<DoubleArray>()
    private val indices = PooledBuffers<IntArray>()
    private val engines = ArrayList<BufferedEngine>()

    internal val arrays: ArraySupplier = object : ArraySupplier {
        override fun take(size: Int): DoubleArray = doubles.take(size, ::DoubleArray)
        override fun takeI32(size: Int): IntArray = indices.take(size, ::IntArray)
        override fun release(buffer: DoubleArray): Unit = doubles.release(buffer)
        override fun release(buffer: IntArray): Unit = indices.release(buffer)
    }

    internal fun bind(engine: KoblasEngine): KoblasEngine {
        val arithmetic = engine.arithmeticEngine
        for (i in engines.indices) if (engines[i].arithmetic === arithmetic) return engines[i]
        return BufferedEngine(arithmetic, arrays).also { engines += it }
    }

    /** Idle floating-point buffers of [size]; an implementation diagnostic tests read to see reuse happen. */
    internal fun available(size: Int): Int = doubles.available(size)

    /** Idle index buffers of [size]; the counterpart of [available]. */
    internal fun availableI32(size: Int): Int = indices.available(size)

    /** Distinct idle floating-point lengths retained, which is what the retention bound is over. */
    internal fun idleLengths(): Int = doubles.idleLengths()

    /** Distinct idle index lengths retained; the counterpart of [idleLengths]. */
    internal fun idleI32Lengths(): Int = indices.idleLengths()
}

/**
 * Primitive buffers lent by exact length and retained under a bound on idle lengths.
 *
 * [MAX_IDLE_LENGTHS] accommodates repeated shapes; evicting the least recently returned length bounds
 * retention across changing shapes. Only lent buffers can be returned, bounding the count within a length.
 *
 * Linear searches avoid a map allocation per length. The idle list is bounded, and the widest schedule
 * holds seven loans at once, so both lists remain small.
 */
private class PooledBuffers<A : Any> {
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
        check(index >= 0) { "released a buffer this workspace did not lend" }
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
