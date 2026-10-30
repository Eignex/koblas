package com.eignex.koblas

import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.assertAgreesWithReference
import com.eignex.koblas.vendor.openBlas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Checks the default's separation of JVM Level 1 kernels from eligible whole-call host products.
 */
class JvmDefaultEngineTest {
    @Test
    fun `the default preserves the owned level one arm with an installed host`() {
        val exact = BuiltinEngines.simd ?: BuiltinEngines.scalar

        assertSame(exact.vectorKernels, koblas.vectorKernels)
        assertSame(exact.indexedSparseKernels, koblas.indexedSparseKernels)
        assertSame(exact.panelKernels, koblas.panelKernels)
        assertTrue(koblas.vendor == null || koblas.vendor?.vendor == Vendor.OneMkl)
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
    @Test
    fun `an explicit jvm host call names the transfer it pays for`() {
        val blas = openBlas() ?: return println(
            "SKIPPED: no CBLAS library installed; the explicit JVM binding was not exercised",
        )
        val a = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 1.0 + it % 5 })
        val c = DenseMatrix.zero(ORDER, ORDER)

        val route = blas.routeOf(
            BlasOperation.Gemm,
            matrices = listOf(a, a, c),
        )
        blas.gemm(1.0, a, false, a, false, 0.0, c)

        assertNotEquals(null, route.adapter, "the JVM binding reported no transfer")
        assertTrue(route.exactlyMeasurable, "an explicit host call is not admissible as a vendor measurement")
        assertTrue(c.values.any { it != 0.0 }, "the explicit binding computed nothing")
    }

    private companion object {
        /** Past the size at which Kotlin/Native's default would hand the same call to a library. */
        const val ORDER = 64
    }
}
