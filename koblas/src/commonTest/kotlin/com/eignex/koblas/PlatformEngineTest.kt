package com.eignex.koblas

import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.internal.dense.PanelWork
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.internal.dense.implementationFor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertSame

/**
 * Checks the default Level 1 arm: Vector API kernels on the JVM, where host calls copy operands,
 * and a vendor on Kotlin/Native, where operands can be pinned.
 */
class PlatformEngineTest {

    @Test
    fun `exact built in engines do not resolve a host library`() {
        assertEquals(null, BuiltinEngines.scalar.vendor)
        assertEquals(null, BuiltinEngines.simd?.vendor)
    }

    // Without the Vector API the owned matrix backends are portable; Kotlin/Native can still compose whole
    // host calls over them, which is checked separately.
    @Test
    fun `the platform selects vector matrix backends where available`() {
        val vector = BuiltinEngines.simd

        if (vector == null) {
            // These are the owned matrix backends; Native whole-call host routing is independent of them.
            assertEquals("scalar-panel", koblas.panelKernels.name)
            assertEquals(BuiltinEngines.scalar.productKernels.name, koblas.productKernels.name)
            assertEquals(BuiltinEngines.scalar.triangularKernels.name, koblas.triangularKernels.name)
            return
        }
        assertSame(vector.vectorKernels, koblas.vectorKernels, "host composition replaced the JVM Level 1 arm")
        assertNotEquals(
            BuiltinEngines.scalar.panelKernels.name,
            koblas.panelKernels.name,
            "an available vector panel backend was not selected",
        )
        assertNotEquals(BuiltinEngines.scalar.productKernels.name, koblas.productKernels.name)
        assertNotEquals(BuiltinEngines.scalar.triangularKernels.name, koblas.triangularKernels.name)
    }

    // A selection is not an execution: the Vector API backend hands a panel shorter than one lane block to
    // the portable body, so this asks the backend which body a width reaches rather than reading its name.
    @Test
    fun `a wide panel reaches the vector body and a short one does not`() {
        val simd = BuiltinEngines.simd ?: return println(
            "SKIPPED: no Vector API engine on this host; the panel bodies were not compared",
        )
        val wide = simd.panelKernels.implementationFor(PanelWork.MultiDot, WIDE, 4)
        val short = simd.panelKernels.implementationFor(PanelWork.MultiDot, 1, 4)

        assertNotEquals(short, wide, "a wide panel and a single row reached the same body")
        assertEquals(PortablePanelKernels.name, short)
    }

    // The JVM never hands Level 1 to a library, so under `-Pkoblas.noSimd=true` on a host with oneMKL the
    // arm is the portable one while a vendor exists; the kernels' own name is what says which was selected.
    @Test
    fun `a wide call reaches whichever accelerated arm this platform has`() {
        if (koblas.vectorKernels.name == PORTABLE) {
            println("SKIPPED: this platform selected the portable kernels; there is no accelerated arm to reach")
            return
        }

        val wide = reached(DenseOperation.Dot, WIDE)

        assertNotEquals(PORTABLE, wide, "a wide dot stayed on the portable kernels")
    }

    /** Below any arm's threshold the call overhead outweighs the arithmetic, so the portable loop runs. */
    @Test
    fun `a single entry call stays with the portable kernels`() {
        assertEquals(PORTABLE, reached(DenseOperation.Dot, 1))
    }

    // `sum` is not a BLAS routine, so no library exports one to reach, but it still vectorises where there
    // are lanes: the assertion is where it cannot go rather than where it must stay.
    @Test
    fun `sum never names a vendor because none exports it`() {
        val vendor = koblas.vendor ?: return
        val vendorArm = "${vendor.vendor.vendorName.lowercase()}-level1"

        assertNotEquals(vendorArm, reached(DenseOperation.Sum, WIDE))
    }

    /**
     * Native Level 1 crossovers differ by operation: at 96 a dot has crossed and a norm has not.
     * `dnrm2` rescales per element while the portable norm first tries a plain sum of squares.
     */
    @Test
    fun `each operation crosses to the vendor at its own measured width`() {
        val vendor = koblas.vendor ?: return skipped("no library on this host")
        val arm = "${vendor.vendor.vendorName.lowercase()}-level1"
        if (koblas.vectorKernels.name != arm) return skipped("this platform keeps Level 1 off the library")

        assertEquals(PORTABLE, reached(DenseOperation.Dot, 32), "a dot at 32 is still the loop's")
        assertEquals(arm, reached(DenseOperation.Dot, 96), "a dot at 96 has crossed")
        assertEquals(PORTABLE, reached(DenseOperation.Nrm2, 96), "the robust norm crosses later")
        assertEquals(PORTABLE, reached(DenseOperation.Iamax, 96), "and so does the index search")
        assertEquals(arm, reached(DenseOperation.Nrm2, 128), "both have crossed by 128")
        assertEquals(arm, reached(DenseOperation.Iamax, 128))
    }

    private fun reached(operation: DenseOperation, length: Int): String? =
        koblas.vectorKernels.implementationFor(operation, length)

    private fun skipped(why: String) {
        println("SKIPPED: $why; the per-operation crossovers were not exercised here")
    }

    private companion object {
        const val PORTABLE = "scalar"

        /** Wider than any arm's crossover, so the accelerated one is reached if there is one. */
        const val WIDE = 1 shl 16
    }
}
