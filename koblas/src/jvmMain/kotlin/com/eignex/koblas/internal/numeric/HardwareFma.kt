package com.eignex.koblas.internal.numeric

import com.sun.management.HotSpotDiagnosticMXBean
import java.lang.management.ManagementFactory

/**
 * Whether HotSpot enables hardware fused multiply-add.
 *
 * Without the instruction, the Vector API preserves single rounding through expensive exact
 * arithmetic per lane. Kernels use separate multiply and add operations instead. HotSpot's flag
 * reflects the compiler's CPU detection and also respects an explicitly disabled `UseFMA`.
 */
internal val hardwareFusedMultiplyAdd: Boolean = hardwareFmaAvailable(hotSpotUseFma())

/**
 * Accept only HotSpot's enabled `UseFMA` value. Unknown flags or runtimes fall back to unfused
 * arithmetic: a false positive would invoke the expensive software FMA fallback.
 *
 * Separate parsing allows both answers to be tested in one process.
 */
internal fun hardwareFmaAvailable(useFma: String?): Boolean = useFma == "true"

/** HotSpot's `UseFMA`, or null on a runtime that does not publish it. */
private fun hotSpotUseFma(): String? = runCatching {
    ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean::class.java).getVMOption("UseFMA").value
}.getOrNull()
