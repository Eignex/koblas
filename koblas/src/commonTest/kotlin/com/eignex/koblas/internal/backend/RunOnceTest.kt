package com.eignex.koblas.internal.backend

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RunOnceTest {

    @Test
    fun `an action runs once however many times it is asked for`() {
        val once = RunOnce()
        var passes = 0

        repeat(3) { once.run { passes++ } }

        assertEquals(1, passes, "the action ran again after it had run")
    }

    /** What a discovery pass gets when it probes a provider that asks for discovery in turn. */
    @Test
    fun `a call from inside the action does not start a second pass`() {
        val once = RunOnce()
        var passes = 0

        once.run {
            passes++
            once.run { passes++ }
        }

        assertEquals(1, passes, "the pass restarted itself from inside")
    }

    @Test
    fun `a pass that threw is run again by the next caller`() {
        val once = RunOnce()
        var passes = 0

        assertFailsWith<IllegalStateException> {
            once.run {
                passes++
                error("the host library could not be opened")
            }
        }
        once.run { passes++ }

        assertEquals(2, passes, "one failed pass settled the question for the process")
    }
}
