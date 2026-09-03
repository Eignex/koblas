package com.eignex.koblas.internal.workspace

import com.eignex.koblas.Workspace

/**
 * Buffers of one array type, pooled by width, and the whole of [Workspace]'s bookkeeping. It is written
 * over the array type rather than over `DoubleArray` so that an element type added later borrows the same
 * reclamation rules instead of a second copy of them, which would be free to drift from these.
 *
 * @param A the array type lent out, one primitive array type per instance.
 * @param allocate a fresh array of the given width.
 * @param widthOf the width of an array handed back, which is how [release] finds its pool.
 */
internal class ArrayPools<A : Any>(private val allocate: (Int) -> A, private val widthOf: (A) -> Int) {
    private var pools = arrayOfNulls<Pool>(INITIAL_WIDTHS)
    private var poolCount = 0

    /** How many widths are pooled. A test hook, since reclamation is otherwise invisible from outside. */
    val pooledWidths: Int get() = poolCount

    /** Borrows a buffer of [size], which must be [release]d; contents are undefined. */
    fun take(size: Int): A {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        return pool(size).take()
    }

    /** Number of already allocated, idle buffers of [size], without opening a pool or allocating. */
    fun available(size: Int): Int {
        require(size >= 0) { "buffer size must not be negative, got $size" }
        return findPool(size)?.available ?: 0
    }

    /** Returns a buffer from [take] to its pool. */
    fun release(buffer: A) {
        val pool = findPool(widthOf(buffer)) ?: notLent(widthOf(buffer))
        pool.release(buffer)
    }

    /** Pre-allocates [count] buffers of [size], so a loop does not allocate even on its first pass. */
    fun reserve(size: Int, count: Int) {
        require(count >= 0) { "reserve count must not be negative, got $count" }
        pool(size).reserve(count)
    }

    /**
     * The pool for [size], created if this is a new width. Found pools move to the front, so a loop that
     * reuses one width finds it first however many other widths have passed through.
     */
    private fun pool(size: Int): Pool {
        findPool(size)?.let { return it }
        // Reclaimed until the new pool fits under the cap, rather than once when the table happens to be
        // full: the table doubles, so a burst of borrows can push the count past the cap with nothing idle to
        // give back. Stops early when every pool has a borrow out, because those have to stay releasable.
        while (poolCount >= MAX_WIDTHS) {
            if (!dropColdestIdlePool()) break
        }
        if (poolCount == pools.size) pools = pools.copyOf(pools.size * 2)
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

    /** Drops the least recently used pool with nothing lent out, reporting whether it found one. */
    private fun dropColdestIdlePool(): Boolean {
        for (i in poolCount - 1 downTo 0) {
            if (pools[i]?.isIdle() == true) {
                for (j in i until poolCount - 1) pools[j] = pools[j + 1]
                pools[--poolCount] = null
                return true
            }
        }
        return false
    }

    /** Buffers of one width, plus which of them are currently lent out. */
    private inner class Pool(val size: Int) {
        private var buffers = arrayOfNulls<Any>(INITIAL_DEPTH)
        private var lent = BooleanArray(INITIAL_DEPTH)

        val available: Int
            get() {
                var count = 0
                for (i in buffers.indices) if (buffers[i] != null && !lent[i]) count++
                return count
            }

        @Suppress("UNCHECKED_CAST") // every slot this fills comes from allocate, so it holds an A
        fun take(): A {
            for (i in buffers.indices) {
                if (!lent[i]) {
                    lent[i] = true
                    return (buffers[i] ?: allocate(size).also { buffers[i] = it }) as A
                }
            }
            val grown = buffers.size * 2
            buffers = buffers.copyOf(grown)
            lent = lent.copyOf(grown)
            val next = buffers.size / 2
            lent[next] = true
            return allocate(size).also { buffers[next] = it }
        }

        fun release(buffer: A) {
            for (i in buffers.indices) {
                if (buffers[i] === buffer) {
                    lent[i] = false
                    return
                }
            }
            notLent(size)
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
            for (i in 0 until count) if (buffers[i] == null) buffers[i] = allocate(size)
        }
    }

    private companion object {
        const val INITIAL_WIDTHS = 2
        const val INITIAL_DEPTH = 4

        /** Distinct widths one array type keeps before the coldest idle pool is recycled. */
        const val MAX_WIDTHS = 64
    }
}

/** Reports a buffer handed back to a workspace that never lent it, whether or not a pool of its width exists. */
private fun notLent(size: Int): Nothing = error("released a buffer of size $size that this workspace did not lend")
