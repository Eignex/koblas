package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class WorkRuleTest {
    @Test
    fun `work estimates saturate without overflowing shape arithmetic`() {
        assertEquals(Long.MAX_VALUE, saturatedWork(Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE))
        assertEquals(0L, saturatedWork(Int.MAX_VALUE, Int.MAX_VALUE, 0))
        assertEquals(8_000_000_000L, saturatedWork(2000, 2000, 2000))
        assertFailsWith<IllegalArgumentException> { saturatedWork(-1) }
    }

    @Test
    fun `never always and zero minimum remain distinct choices`() {
        assertFalse(WorkRule.Never.accepts(Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE))
        assertTrue(WorkRule.AlwaysEligible.accepts(0))
        assertTrue(WorkRule.Minimum(0).accepts(0))
        assertFalse(WorkRule.Minimum(10).accepts(9))
        assertTrue(WorkRule.Minimum(10).accepts(10))
    }

    @Test
    fun `shape rules distinguish skinny and shallow products from square work`() {
        val rule = WorkRule.Shape(4, 4, 1, 8, 128)
        assertTrue(rule.accepts(8, 8, 2))
        assertFalse(rule.accepts(1, 64, 2))
        assertFalse(rule.accepts(8, 8, 9))
    }

    @Test
    fun `scalar and simd call decisions are independent`() {
        val rules = HostCrossovers(WorkRule.Minimum(128), WorkRule.Never, WorkRule.AlwaysEligible)
        assertTrue(rules.scalarToC.accepts(256))
        assertFalse(rules.simdToC.accepts(256))
        assertTrue(rules.nativeToC.accepts(1))
        val reverse = HostCrossovers(WorkRule.Never, WorkRule.Minimum(32), WorkRule.Never)
        assertFalse(reverse.scalarToC.accepts(64))
        assertTrue(reverse.simdToC.accepts(64))
    }
}
