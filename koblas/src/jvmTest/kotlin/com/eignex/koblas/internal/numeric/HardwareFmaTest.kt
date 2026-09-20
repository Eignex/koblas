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
        // Fusing without the instruction costs three orders of magnitude and declining with it costs a few
        // percent, so an unrecognised answer takes the cheap mistake.
        for (reported in listOf(null, "", " ", "false", "TRUE", "1", "yes", "unknown")) {
            assertFalse(hardwareFmaAvailable(reported), "reported $reported")
        }
    }

    @Test
    fun `the resolved answer agrees with the running virtual machine`() {
        // HotSpot supports the flag, so accepting the conservative default here would pass on every host.
        assertTrue(hardwareFusedMultiplyAdd, "no fused multiply-add resolved on a HotSpot host")
    }
}
