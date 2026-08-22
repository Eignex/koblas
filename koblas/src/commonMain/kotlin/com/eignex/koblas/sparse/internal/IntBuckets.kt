package com.eignex.koblas.sparse.internal

/**
 * Items `0 until size` bucketed by a small non-negative key, as one intrusive doubly-linked list per key.
 * Each item sits in at most one bucket and carries its own links, so add, remove and move are all `O(1)`
 * and no allocation happens after construction. The caller remembers each item's current key.
 *
 * @param size how many items there are.
 * @param keys the largest key, so buckets run `0..keys`.
 * @param initialLowest the starting value of the "no occupied bucket is below this" hint. Seed it at 0 to
 *  scan up from the bottom, or above [keys] to mean nothing is known until the first [add].
 */
internal class IntBuckets(private val size: Int, private val keys: Int, initialLowest: Int) {
    private val head = IntArray(keys + 1) { -1 }
    private val next = IntArray(size) { -1 }
    private val previous = IntArray(size) { -1 }

    /** No occupied bucket is below this. A hint rather than a fact, lowered by [add] and raised by lookups. */
    private var lowest = initialLowest

    fun add(item: Int, key: Int) {
        val first = head[key]
        next[item] = first
        previous[item] = -1
        if (first != -1) previous[first] = item
        head[key] = item
        if (key < lowest) lowest = key
    }

    fun remove(item: Int, key: Int) {
        val before = previous[item]
        val after = next[item]
        if (before == -1) head[key] = after else next[before] = after
        if (after != -1) previous[after] = before
        previous[item] = -1
        next[item] = -1
    }

    fun moveTo(item: Int, from: Int, to: Int) {
        if (from == to) return
        remove(item, from)
        add(item, to)
    }

    fun firstAt(key: Int): Int = head[key]

    fun after(item: Int): Int = next[item]

    /** The smallest occupied key at least [from], or -1 when every bucket from there up is empty. */
    fun smallestFrom(from: Int): Int {
        var key = if (from > lowest) from else lowest
        while (key <= keys && head[key] == -1) key++
        if (key > keys) return -1
        if (from <= lowest) lowest = key
        return key
    }

    /** Removes and returns the member with the smallest key; the caller guarantees one exists. */
    fun removeSmallest(): Int {
        while (lowest <= keys && head[lowest] == -1) lowest++
        val item = head[lowest]
        remove(item, lowest)
        return item
    }
}
