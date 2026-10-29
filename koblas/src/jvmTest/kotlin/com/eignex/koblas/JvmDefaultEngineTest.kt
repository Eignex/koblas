package com.eignex.koblas

import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
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
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `the default preserves the owned level one arm with an installed host`() {
        val exact = BuiltinEngines.simd ?: BuiltinEngines.scalar

        assertSame(exact.vectorKernels, koblas.vectorKernels)
        assertSame(exact.indexedSparseKernels, koblas.indexedSparseKernels)
        assertSame(exact.panelKernels, koblas.panelKernels)
        assertTrue(koblas.vendor == null || koblas.vendor?.vendor == Vendor.OneMkl)
    }

    @OptIn(KoblasEngineApi::class)
    @Test
    fun `exact jvm engines keep large dense matrix calls on owned kernels`() {
        val product = DenseCall(256, 256, depth = 256)
        val engines = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)

        for (engine in engines) {
            val route = engine.routeOf(DenseMatrixOperation.Gemm, product)

            assertEquals("portable-dense", route.scheduling, "${engine.name} composed a library")
            assertEquals(null, route.host, "${engine.name} named a vendor call")
            assertEquals(null, engine.vendor, "${engine.name} retained a binding")
        }
    }

    @Test
    fun `the default routes eligible products to its retained host`() {
        val product = DenseCall(256, 256, depth = 256)

        val route = koblas.routeOf(DenseMatrixOperation.Gemm, product)

        assertEquals(koblas.vendor?.vendor, route.host?.vendor)
        assertEquals(if (koblas.vendor == null) "portable-dense" else "host-dense", route.scheduling)
    }

    @Test
    fun `a default host route names its native operand transfers`() {
        if (koblas.vendor == null) return println("SKIPPED: no eligible JVM host composition")

        val route = koblas.routeOf(DenseMatrixOperation.Gemm, DenseCall(256, 256, depth = 256))

        val adapter = requireNotNull(route.host?.adapter)
        assertTrue(route.components.any { it.contains(adapter) }, "default route omitted its operand transfers")
    }

    @Test
    fun `deep whole products can use the host while borrowed products stay owned`() {
        val whole = koblas.routeOf(DenseMatrixOperation.Gemm, DenseCall(16, 16, depth = 4096))
        val borrowed = koblas.routeOf(DenseMatrixOperation.Gemm, DenseCall(16, 16, depth = 4096, borrowed = true))

        assertEquals(koblas.vendor?.vendor, whole.host?.vendor)
        assertEquals(null, borrowed.host)
        assertEquals("portable-dense", borrowed.scheduling)
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

    // A route says which schedule was chosen; this says the chosen one computes.
    @OptIn(KoblasEngineApi::class)
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
