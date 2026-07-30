package com.eignex.koblas

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The open-addressed int-to-double map backing the sparse kernels' scatter buffers.
 *
 * Linear probing with backward-shift deletion is the part that repays testing: removing an entry from
 * the middle of a probe chain must not strand the entries that probed past it, and a table that grows
 * has to rehash every key. The randomized comparison against a HashMap covers the interleavings that
 * hand-written cases do not reach.
 */
class IntDoubleMapTest {

    @Test
    fun `put overwrites an existing key without growing the map`() {
        val m = MutableIntDoubleMap()
        assertEquals(0, m.size)
        assertEquals(-1.0, m.getOrDefault(3, -1.0))
        m.put(3, 1.5)
        assertEquals(1.5, m.getOrDefault(3, -1.0))
        assertEquals(1, m.size)
        m.put(3, 2.5)
        assertEquals(2.5, m.getOrDefault(3, -1.0))
        assertEquals(1, m.size)
    }

    @Test
    fun `slotOf locates a key and valueAt reads it back`() {
        val m = MutableIntDoubleMap()
        m.put(7, 4.25)
        val slot = m.slotOf(7)
        assertTrue(slot >= 0)
        assertEquals(4.25, m.valueAt(slot))
        assertEquals(-1, m.slotOf(8))
    }

    @Test
    fun `negative and boundary keys are storable`() {
        val m = MutableIntDoubleMap()
        val keys = intArrayOf(Int.MIN_VALUE, -1, 0, 1, Int.MAX_VALUE)
        for ((i, k) in keys.withIndex()) m.put(k, i.toDouble())
        for ((i, k) in keys.withIndex()) assertEquals(i.toDouble(), m.getOrDefault(k, Double.NaN))
        assertEquals(keys.size, m.size)
    }

    @Test
    fun `remove reports whether the key was present and deletes it`() {
        val m = MutableIntDoubleMap()
        m.put(1, 1.0)
        m.put(2, 2.0)
        assertTrue(m.remove(1))
        assertFalse(m.remove(1))
        assertEquals(-1, m.slotOf(1))
        assertEquals(2.0, m.getOrDefault(2, Double.NaN))
        assertEquals(1, m.size)
    }

    @Test
    fun `growth preserves every entry`() {
        val m = MutableIntDoubleMap()
        for (k in 0 until 1000) m.put(k * 3, k.toDouble())
        assertEquals(1000, m.size)
        for (k in 0 until 1000) assertEquals(k.toDouble(), m.getOrDefault(k * 3, Double.NaN))
        assertEquals(-1.0, m.getOrDefault(1, -1.0))
    }

    @Test
    fun `backward-shift deletion keeps colliding keys reachable`() {
        // Sequential keys with a small table force probe chains; deleting from the middle of a
        // chain must not strand the entries that probed past the deleted slot.
        val m = MutableIntDoubleMap(4)
        for (k in 0 until 64) m.put(k, k.toDouble())
        for (k in 0 until 64 step 2) assertTrue(m.remove(k))
        for (k in 1 until 64 step 2) assertEquals(k.toDouble(), m.getOrDefault(k, Double.NaN), "key $k lost")
        for (k in 0 until 64 step 2) assertEquals(-1, m.slotOf(k))
        assertEquals(32, m.size)
    }

    @Test
    fun `forEach visits every entry exactly once`() {
        val m = MutableIntDoubleMap()
        for (k in 0 until 100) m.put(k * 7 - 50, k.toDouble())
        val seen = HashMap<Int, Double>()
        m.forEach { k, v -> assertEquals(null, seen.put(k, v), "key $k visited twice") }
        assertEquals(100, seen.size)
        for (k in 0 until 100) assertEquals(k.toDouble(), seen[k * 7 - 50])
    }

    @Test
    fun `scaleValues multiplies every value in place`() {
        val m = MutableIntDoubleMap()
        for (k in 0 until 10) m.put(k, k.toDouble())
        m.scaleValues(0.5)
        for (k in 0 until 10) assertEquals(k * 0.5, m.getOrDefault(k, Double.NaN))
    }

    @Test
    fun `randomized operations agree with a HashMap reference`() {
        val rng = Random(20260727)
        val m = MutableIntDoubleMap()
        val ref = HashMap<Int, Double>()
        repeat(20_000) {
            val key = rng.nextInt(-200, 200)
            when (rng.nextInt(3)) {
                0 -> {
                    val v = rng.nextDouble()
                    m.put(key, v)
                    ref[key] = v
                }

                1 -> assertEquals(ref.remove(key) != null, m.remove(key))

                else -> assertEquals(ref[key] ?: Double.NaN, m.getOrDefault(key, Double.NaN))
            }
            assertEquals(ref.size, m.size)
        }
        m.forEach { k, v -> assertEquals(ref[k], v) }
    }
}
