package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.MatrixWindow
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * A layer that reports a route other than the one it takes.
 *
 * This is what the enforcement exists to catch. Every field a benchmark would read to justify a timing is
 * under the fixture's control, so a check that trusts any single one of them passes here.
 */
private class MisroutedBlas(
    private val delegate: VendorBlas,
    private val claimed: CallRoute,
    override val directlyImplemented: Set<VendorOperation> = delegate.directlyImplemented,
) : VendorBlas by delegate {
    override fun routeOf(operation: VendorOperation, matrices: List<MatrixWindow>): CallRoute = claimed
}

class VendorRouteTest {
    private val storage = DoubleArray(400) { (it % 7).toDouble() + 1.0 }

    private fun columnMajor(rows: Int, columns: Int, offset: Int = 0) =
        MatrixWindow(storage, rows, columns, offset = offset, rowStride = 1, columnStride = 20)

    @Test
    fun `a direct call names the vendor its entry point and the transfer it paid`() = withVendor { blas ->
        val a = columnMajor(3, 4)
        val b = columnMajor(4, 5, offset = 100)
        val c = columnMajor(3, 5, offset = 200)

        val route = blas.routeOf(VendorOperation.Gemm, listOf(a, b, c))

        assertEquals(RouteKind.Direct, route.kind)
        assertEquals(blas.vendor, route.vendor)
        assertEquals("cblas_dgemm", route.entryPoint)
        assertEquals("native buffer transfer", route.adapter)
        assertTrue(route.exactlyMeasurable)
        assertNull(route.reason)
    }

    @Test
    fun `a route admits a copy it had to make`() = withVendor { blas ->
        val a = MatrixWindow(storage, 3, 4, offset = 0, rowStride = 2, columnStride = 40)
        val b = columnMajor(4, 5, offset = 100)
        val c = columnMajor(3, 5, offset = 200)

        val route = blas.routeOf(VendorOperation.Gemm, listOf(a, b, c))

        assertEquals(RouteKind.Direct, route.kind)
        assertEquals("native buffer transfer plus packed copy", route.adapter)
        assertTrue(route.exactlyMeasurable, "staging is part of reaching the vendor, not a different algorithm")
    }

    @Test
    fun `an operation the vendor does not export reports composition rather than its own name`() {
        val library = JvmVendorLibrary.open(Vendor.OpenBlas)
        if (library == null) {
            println("SKIPPED: no CBLAS library installed; composed routing was not verified on this host")
            return
        }
        val blas = JvmVendorBlas(library, suppressed = setOf(VendorOperation.Gemmt))
        val c = MatrixWindow(
            storage,
            4,
            4,
            offset = 200,
            rowStride = 1,
            columnStride = 20,
            structure = MatrixStructure.SymmetricLower,
        )

        val route = blas.routeOf(VendorOperation.Gemmt, listOf(columnMajor(4, 4), columnMajor(4, 4, 100), c))

        assertEquals(RouteKind.Composed, route.kind)
        assertNull(route.entryPoint, "a composed call resolved no entry point of its own")
        assertEquals("cblas_dgemm plus triangle copy", route.adapter)
        assertTrue(!route.exactlyMeasurable, "a composed call is not an exact measurement of gemmt")
    }

    @Test
    fun `the composed path is rejected as an exact arm and the direct one is admitted`() {
        val library = JvmVendorLibrary.open(Vendor.OpenBlas)
        if (library == null) {
            println("SKIPPED: no CBLAS library installed; exact-arm enforcement was not verified on this host")
            return
        }
        val direct = JvmVendorBlas(library)
        val composed = JvmVendorBlas(library, suppressed = setOf(VendorOperation.Gemmt))
        val operands = listOf(columnMajor(4, 4), columnMajor(4, 4, 100), columnMajor(4, 4, 200))

        assertNull(exactArmRejection(direct, VendorOperation.Gemm, operands))
        val rejection = exactArmRejection(composed, VendorOperation.Gemmt, operands)
        assertNotNull(rejection)
        assertTrue("does not implement" in rejection, "unexpected reason: $rejection")
    }

    @Test
    fun `a fixture claiming a direct route for an operation it does not implement is caught`() = withVendor { blas ->
        val fixture = MisroutedBlas(
            blas,
            CallRoute(VendorOperation.Gemmt, RouteKind.Direct, blas.vendor, "cblas_dgemmt", null, null),
            directlyImplemented = blas.directlyImplemented - VendorOperation.Gemmt,
        )

        val rejection = exactArmRejection(fixture, VendorOperation.Gemmt)

        assertNotNull(rejection, "a direct-looking route for an unimplemented operation was admitted")
        assertTrue("does not implement" in rejection, "unexpected reason: $rejection")
    }

    @Test
    fun `a fixture claiming the wrong vendor is caught`() = withVendor { blas ->
        val other = Vendor.entries.first { it != blas.vendor }
        val fixture = MisroutedBlas(
            blas,
            CallRoute(VendorOperation.Gemm, RouteKind.Direct, other, "cblas_dgemm", null, null),
        )

        val rejection = exactArmRejection(fixture, VendorOperation.Gemm)

        assertNotNull(rejection, "a route naming another vendor was admitted")
        assertTrue("not ${blas.vendor.vendorName}" in rejection, "unexpected reason: $rejection")
    }

    @Test
    fun `a fixture claiming the wrong entry point is caught`() = withVendor { blas ->
        val fixture = MisroutedBlas(
            blas,
            CallRoute(VendorOperation.Gemm, RouteKind.Direct, blas.vendor, "cblas_dsymm", null, null),
        )

        val rejection = exactArmRejection(fixture, VendorOperation.Gemm)

        assertNotNull(rejection, "a route resolving a different symbol was admitted")
        assertTrue("cblas_dsymm" in rejection, "unexpected reason: $rejection")
    }

    @Test
    fun `a delegated route is never an exact measurement`() = withVendor { blas ->
        val fixture = MisroutedBlas(
            blas,
            CallRoute(VendorOperation.Gemm, RouteKind.Delegated, blas.vendor, "cblas_dgemm", "scalar", null),
        )

        val rejection = exactArmRejection(fixture, VendorOperation.Gemm)

        assertNotNull(rejection, "a delegated route was admitted as an exact arm")
        assertTrue("delegated" in rejection, "unexpected reason: $rejection")
    }

    @Test
    fun `a vendor that is not installed resolves to nothing rather than to another library`() {
        val absent = Vendor.entries.filter { openVendorBlas(it) == null }

        for (vendor in absent) assertNull(openVendorBlas(vendor), "${vendor.vendorName} resolved unexpectedly")
        val installed = Vendor.entries.mapNotNull { openVendorBlas(it) }
        for (blas in installed) {
            assertEquals(
                blas.vendor,
                blas.routeOf(VendorOperation.Gemm, emptyList()).vendor,
                "a resolved library reported a different vendor than the one requested",
            )
        }
    }
}
