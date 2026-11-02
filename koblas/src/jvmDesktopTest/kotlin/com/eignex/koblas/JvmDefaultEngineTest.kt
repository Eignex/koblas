package com.eignex.koblas

import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.assertAgreesWithReference
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Checks the default's separation of JVM Level 1 kernels from eligible whole-call host products.
 */
class JvmDefaultEngineTest {
    @Test
    fun `the default preserves the owned level one arm with an installed host`() {
        val exact = BuiltinEngines.simd ?: BuiltinEngines.scalar

        assertEquals(
            exact.implementationFor(EngineImplementation.Dot, 1024),
            koblas.implementationFor(EngineImplementation.Dot, 1024),
        )
        assertEquals(exact.tileRows, koblas.tileRows)
        assertEquals(exact.tileColumns, koblas.tileColumns)
        assertTrue(koblas.vendor == null || koblas.vendor == Vendor.OneMkl)
    }

    @Test
    fun `exact jvm engines retain no host binding`() {
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            assertEquals(null, engine.vendor, "${engine.name} retained a binding")
        }
    }

    @Test
    fun `an eligible default product preserves an aliased input buffer`() {
        val order = 128
        val a = DenseMatrix.wrap(order, order, DoubleArray(order * order) { (it % 19 - 9) / 17.0 })
        val b = DenseMatrix.wrap(order, order, DoubleArray(order * order) { (it % 13 - 6) / 11.0 })
        val expected = DenseMatrix.wrap(order, order, a.values.copyOf())
        ReferenceBlas.gemm(0.875, a, false, b, true, -0.25, expected)

        koblas.gemm(0.875, a, false, b, true, -0.25, a, Workspace())

        assertAgreesWithReference(expected.values, a.values, "aliased default host product")
    }

    @Test
    fun `every jvm arm computes a dense product without any library`() {
        val a = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 1.0 + it % 7 })
        val expected = DenseMatrix.zero(ORDER, ORDER)
        ReferenceBlas.gemm(1.0, a, false, a, false, 0.0, expected)

        for (engine in listOfNotNull(koblas, BuiltinEngines.scalar, BuiltinEngines.simd)) {
            val c = DenseMatrix.zero(ORDER, ORDER)
            engine.gemm(1.0, a, false, a, false, 0.0, c)

            assertAgreesWithReference(expected.values, c.values, "${engine.name} product")
        }
    }

    // Every operand is copied into native memory to reach the library, and the route says so, which is what
    // keeps a JVM vendor arm honest about what it is timing.

    private companion object {
        /** Past the size at which Kotlin/Native's default would hand the same call to a library. */
        const val ORDER = 64
    }
}
