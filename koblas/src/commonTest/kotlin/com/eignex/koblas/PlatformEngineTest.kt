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
    /**
     * Asks the arm itself, not what the host happens to have installed.
     *
     * A vendor being present does not make the JVM's Level 1 arm accelerated: the JVM deliberately never
     * routes Level 1 to a library, so under `-Pkoblas.noSimd=true` on a host with oneMKL the engine is the
     * portable one while a vendor exists. The kernels' own name is what says which arm was selected.
     */
    @Test
    fun `a wide call reaches whichever accelerated arm this platform has`() {
        if (koblas.vectorKernels.name == PORTABLE) {
            println("SKIPPED: this platform selected the portable kernels; there is no accelerated arm to reach")
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

    /**
     * The measured crossovers differ by operation, and the routing has to differ with them.
     *
     * One width is enough to catch the mistake this guards against, which is a single constant creeping back:
     * at 512 a dot has crossed and a norm has not, because `dnrm2` rescales for overflow safety per element
     * while the portable kernel tries the plain sum of squares first. Only the Native arm routes Level 1 to a
     * library at all, so everywhere else this says so rather than asserting the JVM into the same shape.
     */
    @Test
    fun `each operation crosses to the vendor at its own measured width`() {
        val vendor = koblas.vendor ?: return skipped("no library on this host")
        val arm = "${vendor.vendor.vendorName.lowercase()}-level1"
        if (koblas.vectorKernels.name != arm) return skipped("this platform keeps Level 1 off the library")

        assertEquals(PORTABLE, koblas.explain(DenseOperation.Dot, 256), "a dot at 256 is still the loop's")
        assertEquals(arm, koblas.explain(DenseOperation.Dot, 512), "a dot at 512 has crossed")
        assertEquals(arm, koblas.explain(DenseOperation.Iamax, 256), "the index search crosses earliest")
        assertEquals(PORTABLE, koblas.explain(DenseOperation.Nrm2, 512), "the robust norm crosses latest")
        assertEquals(arm, koblas.explain(DenseOperation.Nrm2, 768), "and has crossed by 768")
    }

    private fun skipped(why: String) {
        println("SKIPPED: $why; the per-operation crossovers were not exercised here")
    }

    private companion object {
        const val PORTABLE = "scalar"

        /** Wider than any arm's crossover, so the accelerated one is reached if there is one. */
        const val WIDE = 1 shl 16
    }
}
