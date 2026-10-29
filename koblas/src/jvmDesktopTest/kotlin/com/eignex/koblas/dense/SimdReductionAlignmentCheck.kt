package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.internal.jvm25.HotSpotArrayAlignment
import com.eignex.koblas.internal.jvm25.SimdOps
import com.eignex.koblas.testutil.allocation.AllocationProbe
import com.eignex.koblas.testutil.allocation.bytesPerCall

/** Heap layout is fixed at startup, so alignment conformance needs a separate process per collector. */
internal object SimdReductionAlignmentCheck {
    @JvmStatic
    fun main(args: Array<String>) {
        check(BuiltinEngines.simd != null) { "alignment check requires the Vector API arm" }
        val values = DoubleArray(65_560)
        val prefix = HotSpotArrayAlignment.prefix(values, 0, 65_536)
        val knownAlignment = HotSpotArrayAlignment.hasKnownAlignment(values)
        val mode = args.single()
        if (mode == "fallback" || (mode == "object64" && SimdOps.lanes() > 8) ||
            (mode == "zgc" && SimdOps.lanes() > 64)
        ) {
            check(!knownAlignment) { "an unsupported layout reported known alignment" }
            check(prefix == 0) { "an unsupported layout supplied an alignment prefix" }
        } else if (SimdOps.lanes() > 2) {
            check(knownAlignment) { "the selected layout did not report known alignment" }
            check(prefix in 1 until SimdOps.lanes()) { "the selected layout did not exercise alignment" }
        }
        if (mode == "zgc" && SimdOps.lanes() <= 64) {
            check(!HotSpotArrayAlignment.hasKnownAlignment(DoubleArray(32_760))) {
                "a ZGC small array reported known alignment"
            }
            val medium = DoubleArray(32_768)
            check(HotSpotArrayAlignment.hasKnownAlignment(medium)) {
                "a ZGC medium or large array did not report known alignment"
            }
            for (length in 0 until SimdOps.lanes()) {
                check(HotSpotArrayAlignment.prefix(medium, 0, length) in 0..length) {
                    "the alignment prefix exceeded the window"
                }
            }
        }

        SimdOpsTest().`reductions agree with the scalar oracle for windows in large backing arrays`()
        SimdOpsTest().assertFiniteReductionsAgreeWithReference(65_536)
        SimdOpsTest().assertFiniteReductionsAgreeWithReference(65_537)
        val kernels = requireNotNull(BuiltinEngines.simd).vectorKernels
        for (i in values.indices) values[i] = (i % 17 - 8) / 16.0
        val other = DoubleArray(values.size) { (it % 11 - 5) / 8.0 }
        assertAllocationFree("asum") { kernels.asum(values, 0, 65_536) }
        assertAllocationFree("dot") { kernels.dot(values, 0, other, 0, 65_536) }
        assertAllocationFree("iamax") { kernels.iamax(values, 0, 65_536).toDouble() }
        println(
            "reduction alignment conformance passed java=${Runtime.version()} " +
                "prefix=$prefix knownAlignment=$knownAlignment",
        )
    }

    private fun assertAllocationFree(name: String, block: AllocationProbe) {
        val bytes = bytesPerCall(block, warmup = 20_000, iterations = 2_000, windows = 5)
        check(bytes <= 8.0) { "aligned $name allocated $bytes B per call" }
        println("aligned $name allocated $bytes B per call")
    }
}
