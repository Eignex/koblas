package com.eignex.koblas.internal.host

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * The fence is a fence and nothing else: whatever the platform does to hold the owner, the call it wraps has
 * to behave as if it had been made directly.
 */
class NativeReachabilityTest {
    @Test
    fun `a fenced call answers with what it computed`() {
        val owner = Any()

        assertEquals(7, keepingReachable(owner) { 3 + 4 })
    }

    @Test
    fun `a fenced call propagates what it threw`() {
        val failure = assertFailsWith<IllegalStateException> { keepingReachable(Any()) { error("native failure") } }

        assertEquals("native failure", failure.message)
    }
}
