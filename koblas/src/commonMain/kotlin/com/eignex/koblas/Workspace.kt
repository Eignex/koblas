package com.eignex.koblas

import com.eignex.koblas.internal.workspace.ArrayPools

/**
 * Reusable scratch buffers, pooled by width. Every borrow outstanding at the same time gets a different
 * buffer, so nesting is safe. Not thread-safe, deliberately: sharing one across threads corrupts results.
 *
 * A borrow is always exactly the width asked for, so a caller whose sizes vary opens a pool per width.
 * Pools sit in most-recently-used order, and past 64 of them the coldest idle pool is dropped to bound what
 * one workspace holds. A pool with a buffer still lent out is never dropped.
 *
 * [take] lends double-precision buffers, the element type an unqualified borrow means. An element type
 * added later gets a borrow and a return of its own over the same [ArrayPools], and its buffers are pooled
 * apart from these: a pool lends one array type, so the 64 widths are counted per element type.
 */
public class Workspace {
    private val f64 = ArrayPools(::DoubleArray) { it.size }

    /** How many widths are pooled. A test hook, since reclamation is otherwise invisible from outside. */
    internal val pooledWidths: Int get() = f64.pooledWidths

    /** Borrows a vector of [size], which must be [release]d; contents are undefined. Prefer [borrow]. */
    public fun take(size: Int): DoubleArray = f64.take(size)

    /** Returns a buffer from [take] to its pool. */
    public fun release(buffer: DoubleArray): Unit = f64.release(buffer)

    /** Pre-allocates [count] buffers of [size], so a loop does not allocate even on its first pass. */
    public fun reserve(size: Int, count: Int): Unit = f64.reserve(size, count)
}

/** Borrows a vector of [size] for the duration of [block], returning it to [this] afterwards. */
public inline fun <T> Workspace.borrow(size: Int, block: (DoubleArray) -> T): T {
    val buffer = take(size)
    try {
        return block(buffer)
    } finally {
        release(buffer)
    }
}
