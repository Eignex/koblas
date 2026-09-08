package com.eignex.koblas

/**
 * Reusable floating-point and index scratch. Each active borrow gets its own buffer, so nested borrows are
 * safe. A workspace is deliberately not thread-safe.
 */
public class Workspace {
    private val doubles = DoubleBuffers()
    private val indices = IntBuffers()

    @PublishedApi
    internal fun take(size: Int): DoubleArray = doubles.take(size)

    @PublishedApi
    internal fun release(buffer: DoubleArray): Unit = doubles.release(buffer)

    /** Pre-allocates [count] buffers of [size], so a loop does not allocate even on its first pass. */
    public fun reserve(size: Int, count: Int): Unit = doubles.reserve(size, count)

    @PublishedApi
    internal fun takeI32(size: Int): IntArray = indices.take(size)

    @PublishedApi
    internal fun release(buffer: IntArray): Unit = indices.release(buffer)

    /** Pre-allocates [count] integer buffers of [size]. */
    public fun reserveI32(size: Int, count: Int): Unit = indices.reserve(size, count)

    /** Number of reserved floating-point buffers that can satisfy [requirement] without allocation. */
    public fun available(requirement: ScratchRequirement): Int = doubles.available(requirement.size)

    /** Reserves the floating-point buffers described by [requirement]. */
    public fun reserve(requirement: ScratchRequirement): Unit = reserve(requirement.size, requirement.count)
}

/**
 * Borrows a vector of [size] for [block], allocating one when there is no workspace to lend it.
 *
 * Handed back in a finally, so a routine whose kernels throw does not strand the borrow. A buffer never
 * returned is one its pool can neither lend again nor reclaim, which pins that width for the life of the
 * workspace.
 *
 * The receiver is nullable because a workspace is optional everywhere it is taken, and Kotlin cannot carry
 * both receivers under one name: nullability is not part of a JVM signature.
 */
public inline fun <T> Workspace?.borrow(size: Int, block: (DoubleArray) -> T): T {
    val buffer = this?.take(size) ?: DoubleArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

/** Integer counterpart of [borrow]. */
public inline fun <T> Workspace?.borrowI32(size: Int, block: (IntArray) -> T): T {
    val buffer = this?.takeI32(size) ?: IntArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

private class DoubleBuffers {
    private val idle = ArrayList<DoubleArray>()
    private val lent = ArrayList<DoubleArray>()

    fun take(size: Int): DoubleArray {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        val index = idleWithSize(size)
        val buffer = if (index >= 0) idle.removeAt(index) else DoubleArray(size)
        lent += buffer
        return buffer
    }

    fun release(buffer: DoubleArray) {
        val index = lentBuffer(buffer)
        check(index >= 0) { "released a buffer this workspace did not lend" }
        idle += lent.removeAt(index)
    }

    fun reserve(size: Int, count: Int) {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        require(count >= 0) { "reserve count must not be negative, got $count" }
        repeat((count - available(size)).coerceAtLeast(0)) { idle += DoubleArray(size) }
    }

    fun available(size: Int): Int {
        var count = 0
        for (i in idle.indices) if (idle[i].size == size) count++
        return count
    }

    private fun idleWithSize(size: Int): Int {
        for (i in idle.indices) if (idle[i].size == size) return i
        return -1
    }

    private fun lentBuffer(buffer: DoubleArray): Int {
        for (i in lent.indices) if (lent[i] === buffer) return i
        return -1
    }
}

private class IntBuffers {
    private val idle = ArrayList<IntArray>()
    private val lent = ArrayList<IntArray>()

    fun take(size: Int): IntArray {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        val index = idleWithSize(size)
        val buffer = if (index >= 0) idle.removeAt(index) else IntArray(size)
        lent += buffer
        return buffer
    }

    fun release(buffer: IntArray) {
        val index = lentBuffer(buffer)
        check(index >= 0) { "released an index buffer this workspace did not lend" }
        idle += lent.removeAt(index)
    }

    fun reserve(size: Int, count: Int) {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        require(count >= 0) { "reserve count must not be negative, got $count" }
        var idleCount = 0
        for (i in idle.indices) if (idle[i].size == size) idleCount++
        repeat((count - idleCount).coerceAtLeast(0)) { idle += IntArray(size) }
    }

    private fun idleWithSize(size: Int): Int {
        for (i in idle.indices) if (idle[i].size == size) return i
        return -1
    }

    private fun lentBuffer(buffer: IntArray): Int {
        for (i in lent.indices) if (lent[i] === buffer) return i
        return -1
    }
}
