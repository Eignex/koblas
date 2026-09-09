package com.eignex.koblas.sparse.host

import com.eignex.koblas.*
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.SparseLuFactorization
import kotlin.test.*

class SparseDecompositionsRoutingTest {

    /**
     * A backend whose QR comes from a sibling library, so a host can carry its LU without its QR.
     */
    private class SplitLibraryAdapter(override val nativeAvailable: Boolean, private val qrAvailable: Boolean) :
        SparseDecompositionsAdapter() {
        override val name: String get() = "split"

        override fun factorNative(a: SparseMatrix): SparseLuFactorization = error("not reached by routing diagnostics")

        override fun nativeAvailableFor(query: RouteQuery): Boolean =
            if (query is RouteQuery.SparseQr) qrAvailable else super.nativeAvailableFor(query)
    }

    @Test
    fun `a QR whose library is absent routes portably even when the LU resolved`() {
        val adapter = SplitLibraryAdapter(nativeAvailable = true, qrAvailable = false)

        val qr = adapter.route(RouteQuery.SparseQr(1))!!

        assertEquals(BackendExecution.PORTABLE, qr.execution)
        assertEquals(BackendRouteReason.BACKEND_UNAVAILABLE, qr.reason)
    }

    @Test
    fun `the LU still reports natively when its own library resolved`() {
        val adapter = SplitLibraryAdapter(nativeAvailable = true, qrAvailable = false)

        val lu = adapter.route(RouteQuery.SparseLu(1))!!

        assertEquals(BackendExecution.NATIVE, lu.execution)
        assertEquals(BackendRouteReason.NATIVE_ROUTE, lu.reason)
    }

    @Test
    fun `a backend whose routines share one library answers both from it`() {
        val adapter = SplitLibraryAdapter(nativeAvailable = true, qrAvailable = true)

        assertEquals(BackendExecution.NATIVE, adapter.route(RouteQuery.SparseQr(1))!!.execution)
        assertEquals(BackendExecution.NATIVE, adapter.route(RouteQuery.SparseLu(1))!!.execution)
    }
}
