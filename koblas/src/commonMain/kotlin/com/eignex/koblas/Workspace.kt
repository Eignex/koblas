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
 * [take] lends double-precision buffers, the element type an unqualified borrow means. [takeI32] lends
 * integer buffers from independent pools, so the 64 retained widths are counted per element type.
 */
public class Workspace {
    private val f64 = ArrayPools(::DoubleArray) { it.size }
    private val i32 = ArrayPools(::IntArray) { it.size }

    /** How many widths are pooled. A test hook, since reclamation is otherwise invisible from outside. */
    internal val pooledWidths: Int get() = f64.pooledWidths

    /** Borrows a vector of [size], which must be [release]d; contents are undefined. Prefer [borrow]. */
    public fun take(size: Int): DoubleArray = f64.take(size)

    /** Returns a buffer from [take] to its pool. */
    public fun release(buffer: DoubleArray): Unit = f64.release(buffer)

    /** Pre-allocates [count] buffers of [size], so a loop does not allocate even on its first pass. */
    public fun reserve(size: Int, count: Int): Unit = f64.reserve(size, count)

    /** Borrows an integer vector of [size], which must be [release]d; contents are undefined. */
    public fun takeI32(size: Int): IntArray = i32.take(size)

    /** Returns an integer buffer from [takeI32] to its pool. */
    public fun release(buffer: IntArray): Unit = i32.release(buffer)

    /** Pre-allocates [count] integer buffers of [size]. */
    public fun reserveI32(size: Int, count: Int): Unit = i32.reserve(size, count)

    /** Number of reserved idle buffers that can satisfy [requirement] without allocation. */
    public fun available(requirement: ScratchRequirement): Int = when (requirement.kind) {
        ScratchKind.F64 -> f64.available(requirement.size)
        ScratchKind.I32 -> i32.available(requirement.size)
    }

    /** Reserves the buffers described by [requirement]. */
    public fun reserve(requirement: ScratchRequirement): Unit = when (requirement.kind) {
        ScratchKind.F64 -> reserve(requirement.size, requirement.count)
        ScratchKind.I32 -> reserveI32(requirement.size, requirement.count)
    }
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
