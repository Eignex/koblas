// detekt exempts backtick test names and undocumented test functions on the standard test paths only, and
// this is not one of them, so each case below carries its own KDoc.
@file:Suppress("FunctionNaming")

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.engine.*
import com.eignex.koblas.sparse.*
import kotlin.test.Test
import kotlin.test.assertEquals

/** Checks the Native default engine. Missing libraries are reported explicitly. */
class NativeDefaultEngineTest {
    private fun skipped(what: String) {
        println("SKIPPED: no applicable measured host profile or binding; $what was not verified on this host")
    }

    /** Equal runs are staged before the vendor call, without changing arithmetic providers. */
    @Test
    fun `an aliased rotation stays on the selected vendor`() {
        val arm = koblas.implementationFor(EngineImplementation.Rot, LONG_COLUMN)
        if (koblas.providers.none { it.vendor != null && it.name == arm }) return skipped("vendor rotation")
        val values = DoubleArray(LONG_COLUMN) { (it - 17) / 13.0 }
        val expected = values.copyOf()
        BuiltinEngines.scalar.rot(expected, 0, expected, 0, LONG_COLUMN, 0.75, -0.25)

        koblas.rot(values, 0, values, 0, LONG_COLUMN, 0.75, -0.25)

        assertEquals(arm, koblas.implementationFor(EngineImplementation.Rot, LONG_COLUMN))
        assertRotationAgreesWithReference(expected, values)
    }

    private fun assertRotationAgreesWithReference(expected: DoubleArray, actual: DoubleArray) =
        assertClose(expected, actual, "aliased vendor rotation")

    /** The exact portable engine resolves no host binding and provides the availability floor. */
    @Test
    fun `the exact portable engine computes a dense product without a library`() {
        val engine = BuiltinEngines.scalar
        val a = DenseMatrix.wrap(SMALL, SMALL, DoubleArray(SMALL * SMALL) { 1.0 + it })
        val c = DenseMatrix.zero(SMALL, SMALL)
        val expected = DenseMatrix.zero(SMALL, SMALL)
        ReferenceBlas.gemm(1.0, a, false, a, false, 0.0, expected)

        engine.gemm(1.0, a, false, a, false, 0.0, c)

        assertEquals(null, engine.vendor, "the exact portable engine resolved a host library")
        assertClose(expected.values, c.values, "the portable product")
    }

    private companion object {
        const val SMALL = 4

        /** Past the Level 1 crossover, so a contiguous rotation could reach the library. */
        const val LONG_COLUMN = 256
    }
}
