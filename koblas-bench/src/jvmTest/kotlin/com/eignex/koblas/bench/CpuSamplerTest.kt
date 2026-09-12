package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertEquals

class CpuSamplerTest {
    @Test
    fun `missing readings do not contribute to statistics`() {
        val usage = CpuUsage()
        listOf(0.1, -1.0, Double.NaN, 0.3, Double.POSITIVE_INFINITY, 1.1).forEach(usage::add)

        val row = usage.row("baseline", 0.0, 6.0).split(',')

        assertEquals(listOf("baseline", "0.000", "6.000", "6", "2"), row.take(5))
        assertEquals(listOf(20.0, 20.0, 10.0, 30.0), row.drop(5).map(String::toDouble))
    }

    @Test
    fun `unavailable usage has blank statistics`() {
        val usage = CpuUsage()
        usage.add(-1.0)

        val row = usage.row("native", 1.0, 2.0)

        assertEquals("native,1.000,2.000,1,0,,,,", row)
    }
}
