package com.eignex.koblas

/**
 * Reusable scratch for alias-safe matrix operations, taken by every routine that needs temporary storage.
 *
 * A workspace belongs to one invocation at a time. Independent calls may safely use distinct workspaces; calls
 * without one own their temporary storage. Its contents are implementation details and never retain an operand.
 *
 * Buffers are lent for the duration of one nested scope and handed back afterwards, so a routine whose
 * arithmetic throws does not strand a loan. Reuse is by exact length: a workspace lends a buffer of the size
 * asked for or allocates one, which keeps a repeated call over the same shapes allocation-free without making
 * a length mismatch silently read stale entries beyond what it wrote.
 *
 * Retention is bounded. A workspace reused across changing shapes keeps a small number of recently returned
 * lengths and drops the rest, so its memory reflects what the caller is working on rather than everything it
 * has ever worked on. Buffers currently on loan are never dropped, so nested loans remain safe.
 */
public class Workspace {
    private val doubles = PooledBuffers<DoubleArray>()
    private val indices = PooledBuffers<IntArray>()

    @PublishedApi
    @kotlin.jvm.JvmSynthetic
    internal fun take(size: Int): DoubleArray = doubles.take(size, ::DoubleArray)

    @PublishedApi
    @kotlin.jvm.JvmSynthetic
    internal fun release(buffer: DoubleArray): Unit = doubles.release(buffer)

    @PublishedApi
    @kotlin.jvm.JvmSynthetic
    internal fun takeI32(size: Int): IntArray = indices.take(size, ::IntArray)

    @PublishedApi
    @kotlin.jvm.JvmSynthetic
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

/**
 * Borrows a vector of [size] for [block], allocating one when there is no workspace to lend it.
 *
 * Handed back in a `finally`, so a routine whose kernels throw does not strand the borrow. The receiver is
 * nullable because a workspace is optional wherever it is taken, and Kotlin cannot carry both receivers under
 * one name: nullability is not part of a JVM signature.
 *
 * Public because a caller composing its own algorithm on these kernels needs the same bounded scratch the
 * library uses, and a workspace with no way to lend from it would be an object a caller could only
 * pass along. It is `inline`, so the loan and its return are the caller's own code and no lambda survives.
 */
@kotlin.jvm.JvmSynthetic
public inline fun <T> Workspace?.borrow(size: Int, block: (DoubleArray) -> T): T {
    val buffer = this?.take(size) ?: DoubleArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

/** Index counterpart of [borrow]. */
@kotlin.jvm.JvmSynthetic
public inline fun <T> Workspace?.borrowI32(size: Int, block: (IntArray) -> T): T {
    val buffer = this?.takeI32(size) ?: IntArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

/**
 * Buffers of one primitive element type, lent by exact length and retained under an explicit bound.
 *
 * Two bounds, because a workspace is reused in two different ways. A repeated call over one shape asks for the
 * same few lengths every time, so every idle buffer has to survive or the reuse is worthless: that is what
 * [MAX_IDLE_LENGTHS] leaves room for. A caller sweeping changing shapes asks for a new length each time, and
 * retaining every one of them would make a workspace grow with the history of the program rather than with
 * what it is holding: that is what evicting the least recently returned length prevents. Within one length the
 * count is already bounded, because only a buffer this workspace lent can be returned to it.
 *
 * Idle buffers are searched linearly. The widest scheduling here holds seven loans at once, and the idle list
 * is bounded by the two rules above, so both lists stay short enough that a map keyed by length would cost an
 * allocation per distinct size to save a walk over a handful of entries.
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

    /**
     * Distinct idle lengths, which is what the retention bound is over.
     *
     * Counted by scanning rather than by collecting into a set, because returning a buffer asks this on every
     * release and a set would allocate there. The idle list is bounded by the retention rule itself, so the
     * scan is over a handful of entries.
     */
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

    /**
     * Drops the least recently returned length when admitting [size] would exceed the bound.
     *
     * The idle list is in return order, so its first entry names that length. Every buffer of it goes, because
     * a length is what a caller asks for and half of one is of no use to the next call.
     */
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
