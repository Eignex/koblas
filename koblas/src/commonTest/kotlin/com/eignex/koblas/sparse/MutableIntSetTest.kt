package com.eignex.koblas.sparse

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MutableIntSetTest {

    @Test
    fun `add returns a stable slot and does not duplicate`() {
        val s = MutableIntSet()
        val first = s.add(4)
        assertEquals(4, s.keyAt(first))
        assertEquals(first, s.add(4))
        assertEquals(1, s.size)
        assertEquals(first, s.slotOf(4))
        assertEquals(-1, s.slotOf(5))
    }

    @Test
    fun `members occupy the leading slots`() {
        val s = MutableIntSet()
        for (k in 0 until 50) s.add(k * 11)
        val seen = BooleanArray(50)
        for (slot in 0 until s.size) seen[slot] = true
        assertTrue(seen.all { it })
        assertEquals(50, s.size)
    }

    @Test
    fun `removeAt reports the slot it relocated`() {
        val s = MutableIntSet()
        for (k in 0 until 4) s.add(k)
        val slot = s.slotOf(1)
        val moved = s.removeAt(slot)
        assertEquals(3, moved, "the last member should fill the gap")
        assertEquals(3, s.size)
        assertEquals(-1, s.slotOf(1))
        assertEquals(slot, s.slotOf(s.keyAt(slot)))
    }

    @Test
    fun `removing the last member relocates nothing`() {
        val s = MutableIntSet()
        s.add(9)
        assertEquals(-1, s.removeAt(s.slotOf(9)))
        assertEquals(0, s.size)
    }

    @Test
    fun `negative and boundary keys are storable`() {
        val s = MutableIntSet()
        val keys = intArrayOf(Int.MIN_VALUE, -1, 0, 1, Int.MAX_VALUE)
        for (k in keys) s.add(k)
        for (k in keys) assertTrue(s.slotOf(k) >= 0, "key $k lost")
        assertEquals(keys.size, s.size)
    }

    @Test
    fun `growth preserves every member`() {
        val s = MutableIntSet()
        for (k in 0 until 1000) s.add(k * 3)
        assertEquals(1000, s.size)
        for (k in 0 until 1000) assertTrue(s.slotOf(k * 3) >= 0, "key ${k * 3} lost")
        assertEquals(-1, s.slotOf(1))
    }

    @Test
    fun `backward-shift deletion keeps colliding keys reachable`() {
        val s = MutableIntSet(4)
        for (k in 0 until 64) s.add(k)
        for (k in 0 until 64 step 2) assertTrue(s.remove(k))
        for (k in 1 until 64 step 2) assertTrue(s.slotOf(k) >= 0, "key $k lost")
        for (k in 0 until 64 step 2) assertEquals(-1, s.slotOf(k))
        assertEquals(32, s.size)
    }

    @Test
    fun `forEach visits every member exactly once`() {
        val s = MutableIntSet()
        for (k in 0 until 100) s.add(k * 7 - 50)
        val seen = HashSet<Int>()
        s.forEach { k -> assertTrue(seen.add(k), "key $k visited twice") }
        assertEquals(100, seen.size)
        for (k in 0 until 100) assertTrue((k * 7 - 50) in seen)
    }

    @Test
    fun `randomized operations agree with a HashSet reference`() {
        val rng = Random(20260811)
        val s = MutableIntSet()
        val ref = HashSet<Int>()
        repeat(20_000) {
            val key = rng.nextInt(-200, 200)
            when (rng.nextInt(3)) {
                0 -> {
                    s.add(key)
                    ref.add(key)
                }

                1 -> assertEquals(ref.remove(key), s.remove(key))

                else -> assertEquals(key in ref, s.slotOf(key) >= 0)
            }
            assertEquals(ref.size, s.size)
        }
        val seen = HashSet<Int>()
        s.forEach { k -> seen.add(k) }
        assertEquals(ref, seen)
    }

    @Test
    fun `removal keeps a parallel value array aligned`() {
        // The invariant MutableIntDoubleMap relies on: after removeAt the slot to key mapping still agrees.
        val rng = Random(4711)
        val s = MutableIntSet()
        var values = DoubleArray(s.entryCapacity)
        val ref = HashMap<Int, Double>()
        repeat(5_000) {
            val key = rng.nextInt(-100, 100)
            if (rng.nextBoolean()) {
                val slot = s.add(key)
                if (values.size < s.entryCapacity) values = values.copyOf(s.entryCapacity)
                val v = rng.nextDouble()
                values[slot] = v
                ref[key] = v
            } else {
                val slot = s.slotOf(key)
                if (slot >= 0) {
                    val moved = s.removeAt(slot)
                    if (moved >= 0) values[slot] = values[moved]
                    ref.remove(key)
                }
            }
        }
        assertEquals(ref.size, s.size)
        s.forEach { k -> assertEquals(ref[k], values[s.slotOf(k)], "value for $k drifted") }
        assertFalse(ref.isEmpty())
    }
}
