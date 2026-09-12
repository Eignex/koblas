package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DenseProfileTest {
    @Test
    fun `conservative profile preserves separate host paths`() {
        val profile = DenseProfiles.resolve(ProfileOverrides { null })
        assertTrue(profile[DenseOperation.Dot].scalarToC.accepts(128))
        assertFalse(profile[DenseOperation.Dot].scalarToC.accepts(127))
        assertFalse(profile[DenseOperation.Dot].simdToC.accepts(Int.MAX_VALUE))
        assertTrue(profile[DenseOperation.Dot].nativeToC.accepts(48))
        assertFalse(profile[DenseOperation.Axpy].scalarToC.accepts(Int.MAX_VALUE))
    }

    @Test
    fun `per operation overrides resolve once without affecting other runtime paths`() {
        val calls = mutableMapOf<String, Int>()
        val values = mutableMapOf("jvm.simd.c.dot.crossover" to "always", "jvm.c.sum.crossover" to "never")
        val profile = DenseProfiles.resolve(
            ProfileOverrides { key ->
                calls[key] = calls.getOrElse(key) { 0 } + 1
                values[key]
            },
        )
        values.clear()
        assertEquals(WorkRule.AlwaysEligible, profile[DenseOperation.Dot].simdToC)
        assertEquals(WorkRule.Never, profile[DenseOperation.Sum].scalarToC)
        assertEquals(WorkRule.Minimum(128), profile[DenseOperation.Dot].scalarToC)
        assertTrue(calls.values.all { it == 1 })
    }
}
