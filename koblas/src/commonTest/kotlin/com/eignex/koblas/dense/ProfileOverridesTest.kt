package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ProfileOverridesTest {
    @Test
    fun `valid overrides resolve once into immutable schedule data`() {
        val calls = mutableMapOf<String, Int>()
        val overrides = ProfileOverrides { key ->
            calls[key] = calls.getOrElse(key) { 0 } + 1
            if (key == "packed.block.rows") "17" else null
        }
        val schedule = overrides.schedule()
        assertEquals(17, schedule.rows)
        assertTrue(calls.values.all { it == 1 })
        assertTrue(overrides.diagnostics.isEmpty())
    }

    @Test
    fun `malformed rules preserve explicit diagnostic and independent defaults`() {
        val overrides = ProfileOverrides { key -> if (key == "scalar") "0" else "-1" }
        assertEquals(WorkRule.Minimum(0), overrides.rule("scalar", WorkRule.Never))
        assertEquals(WorkRule.AlwaysEligible, overrides.rule("simd", WorkRule.AlwaysEligible))
        assertEquals(1, overrides.diagnostics.size)
    }

    @Test
    fun `impossible combined scratch and mask geometries fall back before freezing`() {
        val huge = ProfileOverrides { key ->
            if (key == "packed.block.rows" || key == "packed.block.depth") Int.MAX_VALUE.toString() else null
        }
        assertEquals(BlockSchedule(), huge.schedule())
        assertEquals(1, huge.diagnostics.size)
        val mask = ProfileOverrides { key -> if (key == "triangular.block") "65" else null }
        assertEquals(64, mask.schedule().diagonalBlock)
        assertEquals(1, mask.diagnostics.size)
    }

    @Test
    fun `never and always overrides do not use numeric infinity sentinels`() {
        val overrides = ProfileOverrides { it }
        assertEquals(WorkRule.Never, overrides.rule("never", WorkRule.AlwaysEligible))
        assertEquals(WorkRule.AlwaysEligible, overrides.rule("always", WorkRule.Never))
    }
}
