package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.RouteKind
import com.eignex.koblas.internal.jvm25.host.JvmVendorBlas
import com.eignex.koblas.internal.jvm25.host.JvmVendorLibrary
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class VendorRouteTest {
    private fun matrix(rows: Int, columns: Int, seed: Int = 0) =
        DenseMatrix.wrap(rows, columns, DoubleArray(rows * columns) { ((it + seed) % 7).toDouble() + 1.0 })

    @Test
    fun `a direct call names the vendor its entry point and the transfer it paid`() = withVendor { blas ->
        val a = matrix(3, 4)
        val b = matrix(4, 5, 1)
        val c = matrix(3, 5, 2)

        val route = blas.routeOf(BlasOperation.Gemm, listOf(a, b, c))

        assertEquals(RouteKind.Direct, route.kind)
        assertEquals(blas.vendor, route.vendor)
        assertEquals("cblas_dgemm", route.entryPoint)
        assertEquals("native buffer transfer", route.adapter)
        assertTrue(route.exactlyMeasurable)
        assertNull(route.reason)
    }

    @Test
    fun `an operation the vendor does not export reports composition rather than its own name`() {
        val library = JvmVendorLibrary.open(Vendor.OpenBlas)
        if (library == null) {
            println("SKIPPED: no CBLAS library installed; composed routing was not verified on this host")
            return
        }
        val blas = JvmVendorBlas(library, suppressed = setOf(BlasOperation.Gemmt))

        val route = blas.routeOf(BlasOperation.Gemmt, listOf(matrix(4, 4), matrix(4, 4, 1), matrix(4, 4, 2)))

        assertEquals(RouteKind.Composed, route.kind)
        assertNull(route.entryPoint, "a composed call resolved no entry point of its own")
        assertEquals("cblas_dgemm plus triangle copy", route.adapter)
        assertTrue(!route.exactlyMeasurable, "a composed call is not an exact measurement of gemmt")
    }

    @Test
    fun `a vendor that is not installed resolves to nothing rather than to another library`() {
        val absent = Vendor.entries.filter { openBlas(it) == null }

        for (vendor in absent) assertNull(openBlas(vendor), "${vendor.vendorName} resolved unexpectedly")
        val installed = Vendor.entries.mapNotNull { openBlas(it) }
        for (blas in installed) {
            assertEquals(
                blas.vendor,
                blas.routeOf(BlasOperation.Gemm, emptyList()).vendor,
                "a resolved library reported a different vendor than the one requested",
            )
        }
    }
}
