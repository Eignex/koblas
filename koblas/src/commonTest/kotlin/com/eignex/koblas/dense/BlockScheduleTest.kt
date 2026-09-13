package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class BlockScheduleTest {
    @Test
    fun `triangular blocks do not follow packing groups`() {
        val schedule = BlockSchedule(leftGroup = 3, rightGroup = 5, diagonalBlock = 17, rhsBlock = 11)
        assertEquals(17, schedule.diagonalBlock)
        assertEquals(11, schedule.rhsBlock)
    }

    @Test
    fun `invalid schedules and overflowing scratch sizes are rejected`() {
        assertFailsWith<IllegalArgumentException> { BlockSchedule(rows = 0) }
        assertFailsWith<IllegalArgumentException> { BlockSchedule(diagonalBlock = 65) }
        assertFailsWith<IllegalArgumentException> { BlockSchedule(nativeCallWorkLimit = 0) }
        assertFailsWith<IllegalArgumentException> { BlockSchedule(rows = Int.MAX_VALUE, depth = Int.MAX_VALUE) }
    }
}
