package com.eignex.koblas

/**
 * Reusable scratch for operations that would otherwise allocate their temporaries on every call.
 *
 * Buffers are pooled by width: an operation borrows a vector of the length it needs and returns it, and
 * the workspace hands out a different buffer for every borrow that is outstanding at the same time. That
 * is what makes nesting safe without any central registry — a condition estimate borrowing four vectors
 * and then calling a triangular solve that borrows a fifth simply grows that width's pool to five. One
 * workspace can therefore serve several dimensions at once, and no two call sites can collide by
 * accident.
 *
 * A pool grows on demand and never shrinks, so a loop allocates during its first iterations and nothing
 * afterwards. Use [reserve] to pay that cost up front instead.
 *
 * Borrowed contents are undefined: an operation that needs zeros clears what it uses.
 *
 * **Not thread-safe, deliberately.** A workspace is caller-owned state, so give each solver instance (or
 * each thread) its own; sharing one across concurrent operations corrupts results. Passing none keeps the
 * allocating behaviour, which is always correct.
 *
 * ```
 * val ws = Workspace().apply { reserve(n, count = 5) }
 * val x = DoubleArray(n)
 * repeat(iterations) {
 *     koblas.solveInto(lu, b, x, workspace = ws)
 *     if (koblas.rcond(lu, anorm, ws) < threshold) koblas.factorInto(a, lu)
 * }
 * ```
 */
class Workspace {
    // Few widths in practice (one or two), so a linear scan beats hashing.
    private var pools = arrayOfNulls<Pool>(INITIAL_WIDTHS)
    private var poolCount = 0

    /**
     * Borrows a vector of [size], which must be [release]d. Prefer [borrow], which returns it for you.
     * Contents are undefined.
     */
    fun take(size: Int): DoubleArray {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        return pool(size).take()
    }

    /** Returns a buffer from [take] to its pool. */
    fun release(buffer: DoubleArray) {
        pool(buffer.size).release(buffer)
    }

    /** Pre-allocates [count] buffers of [size], so a loop does not allocate even on its first pass. */
    fun reserve(size: Int, count: Int) {
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
            // Everything is lent out: deepen the pool. Happens during warm-up, then never.
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
            // A buffer this pool never lent out: nothing to reclaim, and silently accepting it would hide
            // a mismatched borrow.
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
inline fun <T> Workspace.borrow(size: Int, block: (DoubleArray) -> T): T {
    val buffer = take(size)
    try {
        return block(buffer)
    } finally {
        release(buffer)
    }
}
