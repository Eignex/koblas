package com.eignex.koblas.sparse

import kotlin.test.Test
import kotlin.test.assertEquals

class IntBucketsTest {

    @Test
    fun `smallestFrom finds a key above the item count`() {
        // The sparse LU builds these with keys above size, so the two bounds are not interchangeable.
        val buckets = IntBuckets(size = 4, keys = 6, initialLowest = 7)
        buckets.add(item = 2, key = 6)
        assertEquals(6, buckets.smallestFrom(0), "a bucket above size must still be reachable")
        assertEquals(6, buckets.smallestFrom(6))
        assertEquals(2, buckets.firstAt(6))
    }

    @Test
    fun `smallestFrom reports an empty range`() {
        val buckets = IntBuckets(size = 4, keys = 6, initialLowest = 7)
        assertEquals(-1, buckets.smallestFrom(0), "nothing has been added yet")
        buckets.add(item = 1, key = 2)
        assertEquals(-1, buckets.smallestFrom(3), "every bucket from 3 up is empty")
        assertEquals(2, buckets.smallestFrom(0))
    }

    @Test
    fun `smallestFrom and removeSmallest agree on the lowest key`() {
        val buckets = IntBuckets(size = 5, keys = 6, initialLowest = 7)
        buckets.add(item = 0, key = 5)
        buckets.add(item = 1, key = 3)
        buckets.add(item = 2, key = 6)
        assertEquals(3, buckets.smallestFrom(0))
        assertEquals(1, buckets.removeSmallest())
        assertEquals(5, buckets.smallestFrom(0))
        assertEquals(0, buckets.removeSmallest())
        assertEquals(6, buckets.smallestFrom(0))
        assertEquals(2, buckets.removeSmallest())
        assertEquals(-1, buckets.smallestFrom(0))
    }

    @Test
    fun `moveTo relocates an item between buckets`() {
        val buckets = IntBuckets(size = 3, keys = 4, initialLowest = 5)
        buckets.add(item = 1, key = 4)
        buckets.moveTo(item = 1, from = 4, to = 1)
        assertEquals(1, buckets.smallestFrom(0))
        assertEquals(-1, buckets.firstAt(4))
        assertEquals(1, buckets.firstAt(1))
    }

    @Test
    fun `after walks a bucket with several members`() {
        val buckets = IntBuckets(size = 3, keys = 3, initialLowest = 0)
        buckets.add(item = 0, key = 2)
        buckets.add(item = 1, key = 2)
        val first = buckets.firstAt(2)
        val second = buckets.after(first)
        assertEquals(setOf(0, 1), setOf(first, second))
        assertEquals(-1, buckets.after(second))
    }
}
