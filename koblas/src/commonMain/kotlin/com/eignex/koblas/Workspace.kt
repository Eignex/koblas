@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

/**
 * Reusable scratch storage for allocation-sensitive operations.
 *
 * Pass the same workspace to repeated calls to reuse their temporary arrays. Nested operations are safe, but
 * concurrent operations must use separate workspaces.
 */
public class Workspace {
    private val doubles = PrimitiveBuffers(::DoubleArray, DoubleArray::size, "buffer")
    private val indices = PrimitiveBuffers(::IntArray, IntArray::size, "index buffer")

    /** How many floating-point sizes are retained. A test hook for otherwise invisible reclamation. */
    internal val pooledWidths: Int get() = doubles.pooledWidths

    /** How many index sizes are retained. A test hook for otherwise invisible reclamation. */
    internal val pooledI32Widths: Int get() = indices.pooledWidths

    @PublishedApi
    @kotlin.jvm.JvmSynthetic
    internal fun take(size: Int): DoubleArray = doubles.take(size)

    @PublishedApi
    @kotlin.jvm.JvmSynthetic
    internal fun release(buffer: DoubleArray): Unit = doubles.release(buffer)

    /** Pre-allocates [count] buffers of [size]. */
    internal fun reserve(size: Int, count: Int): Unit = doubles.reserve(size, count)

    @PublishedApi
    @kotlin.jvm.JvmSynthetic
    internal fun takeI32(size: Int): IntArray = indices.take(size)

    @PublishedApi
    @kotlin.jvm.JvmSynthetic
    internal fun release(buffer: IntArray): Unit = indices.release(buffer)

    /** Pre-allocates [count] integer buffers of [size]. */
    internal fun reserveI32(size: Int, count: Int): Unit = indices.reserve(size, count)

    /** Number of idle floating-point buffers of [size]. An implementation diagnostic for tests. */
    internal fun available(size: Int): Int = doubles.available(size)
}

/**
 * Borrows a vector of [size] for [block], allocating one when there is no workspace to lend it.
 *
 * Handed back in a finally, so a routine whose kernels throw does not strand the borrow. A buffer never
 * returned remains active, so the workspace can neither lend nor reclaim it.
 *
 * The receiver is nullable because a workspace is optional everywhere it is taken, and Kotlin cannot carry
 * both receivers under one name: nullability is not part of a JVM signature.
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

/** Integer counterpart of [borrow]. */
@kotlin.jvm.JvmSynthetic
public inline fun <T> Workspace?.borrowI32(size: Int, block: (IntArray) -> T): T {
    val buffer = this?.takeI32(size) ?: IntArray(size)
    try {
        return block(buffer)
    } finally {
        this?.release(buffer)
    }
}

private class PrimitiveBuffers<A : Any>(
    private val allocate: (Int) -> A,
    private val sizeOf: (A) -> Int,
    private val description: String,
) {
    private val idle = ArrayList<A>()
    private val lent = ArrayList<A>(INITIAL_LENT_CAPACITY)

    val pooledWidths: Int get() = idleWidthCount()

    fun take(size: Int): A {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        val index = idleWithSize(size)
        val buffer = if (index >= 0) idle.removeAt(index) else allocate(size)
        lent += buffer
        return buffer
    }

    fun release(buffer: A) {
        val index = lentBuffer(buffer)
        check(index >= 0) { "released a $description this workspace did not lend" }
        makeRoomFor(sizeOf(buffer))
        idle += lent.removeAt(index)
    }

    fun reserve(size: Int, count: Int) {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        require(count >= 0) { "reserve count must not be negative, got $count" }
        lent.ensureCapacity(lent.size + count)
        val missing = (count - available(size)).coerceAtLeast(0)
        if (missing > 0) makeRoomFor(size)
        repeat(missing) { idle += allocate(size) }
    }

    fun available(size: Int): Int {
        var count = 0
        for (i in idle.indices) if (sizeOf(idle[i]) == size) count++
        return count
    }

    private fun idleWithSize(size: Int): Int {
        for (i in idle.indices) if (sizeOf(idle[i]) == size) return i
        return -1
    }

    private fun lentBuffer(buffer: A): Int {
        for (i in lent.indices) if (lent[i] === buffer) return i
        return -1
    }

    private fun makeRoomFor(size: Int) {
        if (idleWithSize(size) >= 0 || idleWidthCount() < MAX_IDLE_WIDTHS) return
        val oldestSize = sizeOf(idle[0])
        for (i in idle.lastIndex downTo 0) if (sizeOf(idle[i]) == oldestSize) idle.removeAt(i)
    }

    private fun idleWidthCount(): Int {
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
}

/** Distinct idle sizes retained per primitive element type. */
private const val MAX_IDLE_WIDTHS = 64

/** Small reservations must have bookkeeping storage before their first borrow too. */
private const val INITIAL_LENT_CAPACITY = 4
