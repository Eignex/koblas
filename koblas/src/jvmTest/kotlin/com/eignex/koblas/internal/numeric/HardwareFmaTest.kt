package com.eignex.koblas.internal.numeric

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class HardwareFmaTest {
    @Test
    fun `an enabled flag is the one answer that promises the instruction`() {
        assertTrue(hardwareFmaAvailable("true"))
    }

    @Test
    fun `anything but an enabled flag reads as no instruction`() {
        // A runtime that publishes nothing, one that publishes something else, and the flag turned off are
        // all the same answer here. Fusing on a machine without the instruction costs three orders of
        // magnitude, and declining on one that has it costs a few percent, so an unrecognised answer takes
        // the cheap mistake rather than guessing at the expensive one.
        for (reported in listOf(null, "", " ", "false", "TRUE", "1", "yes", "unknown")) {
            assertFalse(hardwareFmaAvailable(reported), "reported $reported")
        }
    }

    @Test
    fun `the resolved answer agrees with the running virtual machine`() {
        // HotSpot runs these tests and supports the flag, so the resolved value is not allowed to be the
        // conservative default here: that would pass on every host and prove only that a boolean exists.
        assertTrue(hardwareFusedMultiplyAdd, "no fused multiply-add resolved on a HotSpot host")
    }
}
