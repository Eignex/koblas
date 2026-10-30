package com.eignex.koblas.bench

import com.eignex.koblas.RouteKind
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.CallRoute
import com.eignex.koblas.vendor.Vendor
import java.lang.reflect.Proxy
import kotlin.test.Test
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ExactArmRejectionTest {
    private fun binding(route: CallRoute, exports: Set<BlasOperation> = setOf(BlasOperation.Gemm)): Blas =
        Proxy.newProxyInstance(Blas::class.java.classLoader, arrayOf(Blas::class.java)) { _, method, _ ->
            when (method.name) {
                "getVendor" -> Vendor.OpenBlas
                "getDirectlyImplemented" -> exports
                "routeOf" -> route
                else -> error("unexpected call ${method.name}")
            }
        } as Blas

    @Test
    fun `a direct matching call is admitted including its transfer`() {
        val route = CallRoute(BlasOperation.Gemm, RouteKind.Direct, Vendor.OpenBlas, "cblas_dgemm", "transfer", null)

        assertNull(exactArmRejection(binding(route), BlasOperation.Gemm))
    }

    @Test
    fun `admission rejects mismatched execution evidence`() {
        val cases = listOf(
            CallRoute(BlasOperation.Gemm, RouteKind.Composed, Vendor.OpenBlas, null, "composition", null) to "composed",
            CallRoute(BlasOperation.Gemm, RouteKind.Delegated, Vendor.OpenBlas, "cblas_dgemm", "scalar", null) to "delegated",
            CallRoute(BlasOperation.Gemm, RouteKind.Direct, Vendor.OneMkl, "cblas_dgemm", null, null) to "not OpenBLAS",
            CallRoute(BlasOperation.Gemm, RouteKind.Direct, Vendor.OpenBlas, "cblas_dsymm", null, null) to "cblas_dsymm",
        )
        for ((route, reason) in cases) {
            val rejection = exactArmRejection(binding(route), BlasOperation.Gemm)

            assertTrue(rejection != null && reason.lowercase() in rejection.lowercase(), "$route: $rejection")
        }
    }

    @Test
    fun `a direct looking route cannot admit an unimplemented operation`() {
        val route = CallRoute(BlasOperation.Gemm, RouteKind.Direct, Vendor.OpenBlas, "cblas_dgemm", null, null)

        val rejection = exactArmRejection(binding(route, emptySet()), BlasOperation.Gemm)

        assertTrue(rejection != null && "does not implement" in rejection)
    }
}
