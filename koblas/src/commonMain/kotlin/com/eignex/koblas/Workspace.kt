package com.eignex.koblas

/**
 * Reusable scratch buffers, pooled by width. Every borrow outstanding at the same time gets a different
 * buffer, so nesting is safe. Not thread-safe, deliberately: sharing one across threads corrupts results.
 */
public class Workspace {
    private var pools = arrayOfNulls<Pool>(INITIAL_WIDTHS)
    private var poolCount = 0

    /** Borrows a vector of [size], which must be [release]d; contents are undefined. Prefer [borrow]. */
    public fun take(size: Int): DoubleArray {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        return pool(size).take()
    }

    /** Returns a buffer from [take] to its pool. */
    public fun release(buffer: DoubleArray) {
        pool(buffer.size).release(buffer)
    }

    /** Pre-allocates [count] buffers of [size], so a loop does not allocate even on its first pass. */
    public fun reserve(size: Int, count: Int) {
        require(count >= 0) { "reserve count must not be negative, got $count" }
        pool(size).reserve(count)
    }

    private fun pool(size: Int): Pool {
        for (i in 0 until poolCount) {
            val existing = pools[i]
            if (existing != null && existing.size == size) return existing
        }
        if (poolCount == pools.size) {
            pools = pools.copyOf(pools.size * 2)
        }
        val fresh = Pool(size)
        pools[poolCount++] = fresh
        return fresh
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
            error("released a buffer of size ${buffer.size} that this workspace did not lend")
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
    }
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
