package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.testutil.allocation.AllocationProbe
import com.eignex.koblas.testutil.allocation.bytesPerCall

/**
 * Exercises Vector casts with Kotlin's null helper deliberately kept out of line.
 *
 * An opaque helper must receive only null, never a live Vector carrier, regardless of C2's inlining budget.
 * The primitive probe interface also keeps measurement callbacks from boxing their results.
 */
internal object VectorBoundaryAllocationCheck {
    @JvmStatic
    fun main(args: Array<String>) {
        for (length in listOf(17, 129, 4_097, 65_537)) {
            val peaked = DoubleArray(length + 3) { if (it == 3) -2.0 else -0.5 }
            val increasing = DoubleArray(length + 3) { -it.toDouble() }
            assertAllocationFree("asum length=$length") { SimdVectorOps.asum(peaked, 3, length) }
            assertAllocationFree("iamax filter length=$length") { SimdVectorOps.iamax(peaked, 3, length).toDouble() }
            assertAllocationFree(
                "iamax reduction length=$length",
            ) { SimdVectorOps.iamax(increasing, 3, length).toDouble() }
        }
    }

    private fun assertAllocationFree(name: String, probe: AllocationProbe) {
        val bytes = bytesPerCall(probe, warmup = 20_000, iterations = 2_000, windows = 8)
        check(bytes <= 8.0) { "$name allocated $bytes B per call" }
        println("$name allocated $bytes B per call")
    }
}
