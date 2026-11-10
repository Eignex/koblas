package com.eignex.koblas.internal.jvm25.simd

import com.sun.management.HotSpotDiagnosticMXBean
import java.lang.management.ManagementFactory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SimdRuntimeTest {
    @Test
    fun `only an enabled flag promises the instruction`() {
        assertTrue(hardwareFmaAvailable("true"))
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
