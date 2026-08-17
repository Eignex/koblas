package com.eignex.koblas

/**
 * Reusable scratch buffers, pooled by width. Every borrow outstanding at the same time gets a different
 * buffer, so nesting is safe. Not thread-safe, deliberately: sharing one across threads corrupts results.
 *
 * A borrow is always exactly the width asked for, so a caller whose sizes vary opens a pool per width.
 * Pools sit in most-recently-used order, and past 64 of them the coldest idle pool is dropped to bound what
 * one workspace holds. A pool with a buffer still lent out is never dropped.
 */
public class Workspace {
    private var pools = arrayOfNulls<Pool>(INITIAL_WIDTHS)
    private var poolCount = 0

    /** How many widths are pooled. A test hook, since reclamation is otherwise invisible from outside. */
    internal val pooledWidths: Int get() = poolCount

    /** Borrows a vector of [size], which must be [release]d; contents are undefined. Prefer [borrow]. */
    public fun take(size: Int): DoubleArray {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        return pool(size).take()
    }

    /** Returns a buffer from [take] to its pool. */
    public fun release(buffer: DoubleArray) {
        val pool = findPool(buffer.size) ?: notLent(buffer)
        pool.release(buffer)
    }

    /** Pre-allocates [count] buffers of [size], so a loop does not allocate even on its first pass. */
    public fun reserve(size: Int, count: Int) {
        require(count >= 0) { "reserve count must not be negative, got $count" }
        pool(size).reserve(count)
    }

    /**
     * The pool for [size], created if this is a new width. Found pools move to the front, so a loop that
     * reuses one width finds it first however many other widths have passed through.
     */
    private fun pool(size: Int): Pool {
        findPool(size)?.let { return it }
        if (poolCount == pools.size) {
            // Below the cap the table simply grows. Once at it, the coldest idle pool is reclaimed, and the
            // table still grows when every pool has a borrow out, since those must stay releasable.
            if (poolCount >= MAX_WIDTHS) dropColdestIdlePool()
            if (poolCount == pools.size) pools = pools.copyOf(pools.size * 2)
        }
        val fresh = Pool(size)
        for (i in poolCount downTo 1) pools[i] = pools[i - 1]
        pools[0] = fresh
        poolCount++
        return fresh
    }

    private fun findPool(size: Int): Pool? {
        for (i in 0 until poolCount) {
            val existing = pools[i]
            if (existing != null && existing.size == size) {
                if (i != 0) moveToFront(i)
                return existing
            }
        }
        return null
    }

    private fun moveToFront(index: Int) {
        val found = pools[index]
        for (i in index downTo 1) pools[i] = pools[i - 1]
        pools[0] = found
    }

    /** Drops the least recently used pool with nothing lent out, and does nothing when every pool is in use. */
    private fun dropColdestIdlePool() {
        for (i in poolCount - 1 downTo 0) {
            if (pools[i]?.isIdle() == true) {
                for (j in i until poolCount - 1) pools[j] = pools[j + 1]
                pools[--poolCount] = null
                return
            }
        }
    }

    /** Buffers of one width, plus which of them are currently lent out. */
    private class Pool(val size: Int) {
        private var buffers = arrayOfNulls<DoubleArray>(INITIAL_DEPTH)
        private var lent = BooleanArray(INITIAL_DEPTH)

        fun take(): DoubleArray {
            for (i in buffers.indices) {
                if (!lent[i]) {
                    lent[i] = true
                    return buffers[i] ?: DoubleArray(size).also { buffers[i] = it }
                }
            }
            val grown = buffers.size * 2
            buffers = buffers.copyOf(grown)
            lent = lent.copyOf(grown)
            val next = buffers.size / 2
            lent[next] = true
            return DoubleArray(size).also { buffers[next] = it }
        }

        fun release(buffer: DoubleArray) {
            for (i in buffers.indices) {
                if (buffers[i] === buffer) {
                    lent[i] = false
                    return
                }
            }
            notLent(buffer)
        }

        /** Whether every buffer here is back, so the pool can be dropped without losing a live borrow. */
        fun isIdle(): Boolean {
            for (i in buffers.indices) if (lent[i]) return false
            return true
        }

        fun reserve(count: Int) {
            if (count > buffers.size) {
                buffers = buffers.copyOf(count)
                lent = lent.copyOf(count)
            }
            for (i in 0 until count) if (buffers[i] == null) buffers[i] = DoubleArray(size)
        }
    }

    private companion object {
        const val INITIAL_WIDTHS = 2
        const val INITIAL_DEPTH = 4

        /** Distinct widths kept before the coldest idle pool is recycled. */
        const val MAX_WIDTHS = 64
    }
}

/** Reports a buffer handed back to a workspace that never lent it, whether or not a pool of its width exists. */
private fun notLent(buffer: DoubleArray): Nothing =
    error("released a buffer of size ${buffer.size} that this workspace did not lend")

/** Borrows a vector of [size] for the duration of [block], returning it to [this] afterwards. */
public inline fun <T> Workspace.borrow(size: Int, block: (DoubleArray) -> T): T {
    val buffer = take(size)
    try {
        return block(buffer)
    } finally {
        release(buffer)
    }
}
