// detekt exempts backtick test names and undocumented test functions on the standard test paths only, and
// this is not one of them, so each case below carries its own KDoc.
@file:Suppress("FunctionNaming")

package com.eignex.koblas

import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.internal.dense.implementationFor
import kotlin.test.Test
import kotlin.test.assertEquals

/** Checks the Native default engine. Missing libraries are reported explicitly. */
class NativeDefaultEngineTest {
    private fun skipped(what: String) {
        println("SKIPPED: no applicable measured host profile or binding; $what was not verified on this host")
    }

    /**
     * Aliased rotations stay portable because reference `drot` reads an entry after storing it,
     * while owned kernels read both inputs before writing. Width alone cannot settle the kernel.
     */
    @Test
    fun `a wide rotation is decided by its operands rather than its width`() {
        val vendor = koblas.vendor ?: return skipped("the rotation's alias-dependent dispatch")
        val arm = "${vendor.vendor.vendorName.lowercase()}-level1"
        if (koblas.vectorKernels.implementationFor(
                EngineImplementation.Dot,
                LONG_COLUMN,
            ) != arm
        ) {
            return skipped("this platform keeps Level 1 off the library")
        }

        assertEquals(null, koblas.vectorKernels.implementationFor(EngineImplementation.Rot, LONG_COLUMN))
    }

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
