package com.eignex.koblas

/**
 * Reusable scratch for alias-safe matrix products and sparse matrix algorithms.
 *
 * A workspace belongs to one invocation at a time. Independent calls may safely use distinct workspaces; calls
 * without one own their temporary storage. Its contents are implementation details and never retain an operand.
 *
 * Buffers are lent for the duration of one nested scope and handed back afterwards, so a routine whose
 * arithmetic throws does not strand a loan. Reuse is by exact length: a workspace lends a buffer of the size
 * asked for or allocates one, which keeps a repeated call over the same shapes allocation-free without making
 * a length mismatch silently read stale entries beyond what it wrote.
 */
public class MatrixWorkspace {
    private val doubles = PooledBuffers<DoubleArray>()
    private val indices = PooledBuffers<IntArray>()

    internal fun leftCopy(values: DoubleArray): DoubleArray = doubles.retained(0, values.size, ::DoubleArray)
        .also { values.copyInto(it) }

    internal fun rightCopy(values: DoubleArray): DoubleArray = doubles.retained(1, values.size, ::DoubleArray)
        .also { values.copyInto(it) }

    internal fun take(size: Int): DoubleArray = doubles.take(size, ::DoubleArray)

    internal fun release(buffer: DoubleArray): Unit = doubles.release(buffer)

    internal fun takeI32(size: Int): IntArray = indices.take(size, ::IntArray)

    internal fun release(buffer: IntArray): Unit = indices.release(buffer)

    /** Idle floating-point buffers of [size]; an implementation diagnostic tests read to see reuse happen. */
    internal fun available(size: Int): Int = doubles.available(size)

    /** Idle index buffers of [size]; the counterpart of [available]. */
    internal fun availableI32(size: Int): Int = indices.available(size)
}

/**
 * Borrows a vector of [size] for [block], allocating one when there is no workspace to lend it.
 *
 * Handed back in a `finally`, so a routine whose kernels throw does not strand the borrow. The receiver is
 * nullable because a workspace is optional wherever it is taken.
 */
internal inline fun <T> MatrixWorkspace?.borrow(size: Int, block: (DoubleArray) -> T): T {
    val buffer = this?.take(size) ?: DoubleArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

/** Index counterpart of [borrow]. */
internal inline fun <T> MatrixWorkspace?.borrowI32(size: Int, block: (IntArray) -> T): T {
    val buffer = this?.takeI32(size) ?: IntArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

/**
 * Buffers of one primitive element type, lent by exact length.
 *
 * Idle buffers are searched linearly because a single invocation holds a handful of loans at once: the widest
 * sparse scheduling here borrows seven. A map keyed by length would cost an allocation per distinct size to
 * save a walk over a list that never grows past that handful.
 */
private class PooledBuffers<A : Any> {
    private val idle = ArrayList<A>()
    private val lent = ArrayList<A>()
    private val retained = HashMap<Int, A>()

    fun take(size: Int, allocate: (Int) -> A): A {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        val index = idleWithSize(size)
        val buffer = if (index >= 0) idle.removeAt(index) else allocate(size)
        lent += buffer
        return buffer
    }

    fun release(buffer: A) {
        val index = lent.indexOfFirst { it === buffer }
        check(index >= 0) { "released a buffer this workspace did not lend" }
        idle += lent.removeAt(index)
    }

    /**
     * A buffer kept under [slot] across calls rather than lent for a scope.
     *
     * Alias staging needs its copy to outlive the borrow scope: the staged operand is read by the whole
     * operation, so the copy is held by slot until a call asks for a different length.
     */
    fun retained(slot: Int, size: Int, allocate: (Int) -> A): A {
        val existing = retained[slot]
        if (existing != null && sizeOf(existing) == size) return existing
        val fresh = allocate(size)
        retained[slot] = fresh
        return fresh
    }

    fun available(size: Int): Int = idle.count { sizeOf(it) == size }

    private fun idleWithSize(size: Int): Int {
        for (i in idle.indices) if (sizeOf(idle[i]) == size) return i
        return -1
    }

    private fun sizeOf(buffer: A): Int = when (buffer) {
        is DoubleArray -> buffer.size
        is IntArray -> buffer.size
        else -> error("unsupported workspace buffer")
    }
}
