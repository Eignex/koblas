package com.eignex.koblas.internal.numeric

import com.eignex.koblas.internal.jvm25.hardwareFmaAvailable
import com.eignex.koblas.internal.jvm25.hardwareFusedMultiplyAdd
import com.sun.management.HotSpotDiagnosticMXBean
import java.lang.management.ManagementFactory
import kotlin.test.Test
import kotlin.test.assertEquals
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
        assertHardwareFmaAgreesWithVm()
    }
}

/** Also checked in the runtime harness, where `UseFMA` is explicitly disabled. */
internal fun assertHardwareFmaAgreesWithVm() {
    val reported = runCatching {
        ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean::class.java).getVMOption("UseFMA").value
    }.getOrNull()

    assertEquals(reported == "true", hardwareFusedMultiplyAdd, "the VM reported UseFMA=$reported")
}
