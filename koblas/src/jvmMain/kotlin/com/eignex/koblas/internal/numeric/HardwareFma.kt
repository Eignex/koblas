package com.eignex.koblas.internal.numeric

import com.sun.management.HotSpotDiagnosticMXBean
import java.lang.management.ManagementFactory

/**
 * Whether this machine has an instruction behind a fused multiply-add.
 *
 * The Vector API offers one everywhere and says nothing about where it is real. Machines without the
 * instruction get a fallback that cannot multiply and then add, since that rounds twice where a fused one
 * rounds once, so it computes the product exactly instead: a `BigDecimal` per operand at unbounded precision,
 * one call per lane. Measured against the scalar loop it is meant to beat, that is three orders of magnitude
 * slower, which is why a kernel asks this question before fusing rather than after.
 *
 * Read from HotSpot rather than guessed from the architecture, because HotSpot decides it by the same CPU
 * detection its compiler uses and publishes the answer. Forcing the flag off reproduces the fallback on a
 * machine that does have the instruction, which is what says the flag governs this and not only [Math.fma].
 */
internal val hardwareFusedMultiplyAdd: Boolean = hardwareFmaAvailable(hotSpotUseFma())

/**
 * Whether [useFma], as HotSpot reports its `UseFMA` flag, promises the instruction.
 *
 * Only the word it writes for an enabled flag counts, so everything else reads as absent: another virtual
 * machine, a runtime linked without `jdk.management`, or a flag that has been retired. The two mistakes are
 * not worth trading evenly. Believing an instruction is present where it is not costs the whole pathology
 * above, while missing one that is present costs a few percent on the lengths that fit in cache.
 *
 * Separate from reading the flag so both answers can be tested in one process, which interrogating a virtual
 * machine's own configuration cannot be.
 */
internal fun hardwareFmaAvailable(useFma: String?): Boolean = useFma == "true"

/** HotSpot's `UseFMA`, or null on a runtime that does not publish it. */
private fun hotSpotUseFma(): String? = runCatching {
    ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean::class.java).getVMOption("UseFMA").value
}.getOrNull()
