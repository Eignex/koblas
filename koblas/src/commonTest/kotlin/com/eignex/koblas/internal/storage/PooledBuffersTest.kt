package com.eignex.koblas.internal.storage

import kotlin.test.Test
import kotlin.test.assertEquals

class PooledBuffersTest {
    @Test
    fun `eviction uses the most recent return of each length`() {
        assertRecentLengthRetained(::DoubleArray)
        assertRecentLengthRetained(::IntArray)
    }

    private fun <A : Any> assertRecentLengthRetained(allocate: (Int) -> A) {
        val pool = PooledBuffers<A>()
        val first = pool.take(1, allocate)
        val second = pool.take(1, allocate)
        pool.release(first)
        pool.release(second)
        for (length in 2..MAX_IDLE_LENGTHS) pool.release(pool.take(length, allocate))
        pool.release(pool.take(1, allocate))

        pool.release(pool.take(MAX_IDLE_LENGTHS + 1, allocate))

        assertEquals(2, pool.available(1), "a recently returned length was evicted")
        assertEquals(0, pool.available(2), "the least recently returned length was retained")
        assertEquals(MAX_IDLE_LENGTHS, pool.idleLengths())
    }
}
