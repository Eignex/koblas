package com.eignex.koblas

import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.PortableEngine
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

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

    @Test
    fun `exact SIMD uses supplied kernels from their intrinsic minimum length`() {
        val simd = BuiltinEngines.simd ?: return
        val leaf = simd.providers.first { it.id == "jvm-simd" && EngineImplementation.Iamax in it.implementations }

        assertEquals(
            leaf.name,
            simd.implementationFor(EngineImplementation.Iamax, leaf.minimumLength(EngineImplementation.Iamax)),
        )
    }

    // Without the Vector API the owned matrix backends are portable; Kotlin/Native can still compose whole
    // host calls over them, which is checked separately.
    @Test
    fun `the platform selects vector matrix backends where available`() {
        val vector = BuiltinEngines.simd

        if (vector == null) {
            // These are the owned matrix backends; Native whole-call host routing is independent of them.
            assertEquals(PortableEngine.name, koblas.implementationFor(PanelWork.RankUpdate, WIDE, 4))
            return
        }
        assertEquals(
            vector.implementationFor(EngineImplementation.Dot, WIDE),
            reached(EngineImplementation.Dot, WIDE),
        )
        assertNotEquals(
            BuiltinEngines.scalar.name,
            koblas.name,
            "an available vector panel backend was not selected",
        )
    }

    // A selection is not an execution: the Vector API backend hands a rank update shorter than one lane block
    // to the portable body, so this asks the backend which body a width reaches rather than reading its name.
    @Test
    fun `a wide panel reaches the vector body and a short one does not`() {
        val simd = BuiltinEngines.simd ?: return println(
            "SKIPPED: no Vector API engine on this host; the panel bodies were not compared",
        )
        val wide = simd.implementationFor(PanelWork.RankUpdate, WIDE, 4)
        val short = simd.implementationFor(PanelWork.RankUpdate, 1, 4)

        assertNotEquals(short, wide, "a wide panel and a single row reached the same body")
        assertEquals(PortableEngine.name, short)
    }

    /** Below any arm's threshold the call overhead outweighs the arithmetic, so the portable loop runs. */
    @Test
    fun `a single entry call stays with the portable kernels`() {
        assertEquals(PORTABLE, reached(EngineImplementation.Dot, 1))
    }

    // `sum` is not a BLAS routine, so no library exports one to reach, but it still vectorises where there
    // are lanes: the assertion is where it cannot go rather than where it must stay.
    @Test
    fun `sum never names a vendor because none exports it`() {
        for (vendor in koblas.providers.filter { it.vendor != null }) {
            assertNotEquals(vendor.name, reached(EngineImplementation.Sum, WIDE))
        }
    }

    private fun reached(operation: EngineImplementation, length: Int): String? =
        koblas.implementationFor(operation, length)

    private companion object {
        const val PORTABLE = "scalar"

        /** Wider than any arm's crossover, so the accelerated one is reached if there is one. */
        const val WIDE = 1 shl 16
    }
}
