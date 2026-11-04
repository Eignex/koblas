package com.eignex.koblas.internal.jvm25.simd

import com.sun.management.HotSpotDiagnosticMXBean
import jdk.incubator.vector.DoubleVector
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

/**
 * One multiply-add, fused where [hardwareFusedMultiplyAdd] found the instruction and two operations
 * where it did not.
 *
 * Inline because a helper returning a [DoubleVector] that the JIT declines to inline makes the vector a
 * heap object, allocating per loop iteration or tile depth step. Kotlin inlining keeps the vector in the
 * caller and lets the JIT fold the hardware flag so only one arithmetic form reaches the loop.
 */
@Suppress("NOTHING_TO_INLINE")
internal inline fun multiplyAdd(x: DoubleVector, y: DoubleVector, accumulator: DoubleVector): DoubleVector =
    if (hardwareFusedMultiplyAdd) x.fma(y, accumulator) else x.mul(y).add(accumulator)
