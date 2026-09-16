package com.eignex.koblas

import com.eignex.koblas.dense.DenseOperation
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

/**
 * Which Level 1 arm the default engine selected, which nothing else here would notice.
 *
 * Each platform prefers a different one: the Vector API kernels on the JVM, where reaching a foreign library
 * copies both operands first, and the vendor on Kotlin/Native, which pins them instead and has no Vector API.
 * Every other test asks only for a correct answer, and both arms give that, so a default that quietly kept
 * running portable Kotlin everywhere would pass all of them.
 */
class PlatformEngineTest {
    @Test
    fun `a wide call reaches whichever accelerated arm this platform has`() {
        val accelerated = koblas.vendor != null || koblas.vectorKernels.name != PORTABLE
        if (!accelerated) {
            println("SKIPPED: neither a Vector API module nor a CBLAS library here; no accelerated arm to reach")
            return
        }

        val wide = koblas.explain(DenseOperation.Dot, WIDE)

        assertNotEquals(PORTABLE, wide, "a wide dot stayed on the portable kernels")
    }

    /** Below any arm's threshold the call overhead outweighs the arithmetic, so the portable loop runs. */
    @Test
    fun `a single entry call stays with the portable kernels`() {
        assertEquals(PORTABLE, koblas.explain(DenseOperation.Dot, 1))
    }

    /**
     * `sum` is not a BLAS routine, so no library exports one to reach.
     *
     * It still vectorises where there are lanes to use, which is why this says where it cannot go rather than
     * where it must stay.
     */
    @Test
    fun `sum never names a vendor because none exports it`() {
        val vendor = koblas.vendor ?: return
        val vendorArm = "${vendor.vendor.vendorName.lowercase()}-level1"

        assertNotEquals(vendorArm, koblas.explain(DenseOperation.Sum, WIDE))
    }

    private companion object {
        const val PORTABLE = "scalar"

        /** Wider than any arm's crossover, so the accelerated one is reached if there is one. */
        const val WIDE = 1 shl 16
    }
}
