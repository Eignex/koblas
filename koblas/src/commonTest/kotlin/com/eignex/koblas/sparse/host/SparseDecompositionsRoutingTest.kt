package com.eignex.koblas.sparse.host

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64SparseLuFactorization
import kotlin.test.*

class SparseDecompositionsRoutingTest {

    /**
     * A backend whose QR comes from a sibling library, so a host can carry its LU without its QR.
     */
    private class SplitLibraryAdapter(override val nativeAvailable: Boolean, private val qrAvailable: Boolean) :
        F64SparseDecompositionsAdapter() {
        override val name: String get() = "split"

        override fun factorNative(a: F64SparseMatrix): F64SparseLuFactorization =
            error("not reached by routing diagnostics")

        override fun nativeAvailableFor(query: F64RouteQuery): Boolean =
            if (query is F64RouteQuery.SparseQr) qrAvailable else super.nativeAvailableFor(query)
    }

    @Test
    fun `a QR whose library is absent routes portably even when the LU resolved`() {
        val adapter = SplitLibraryAdapter(nativeAvailable = true, qrAvailable = false)

        val qr = adapter.route(F64RouteQuery.SparseQr(1))!!

        assertEquals(BackendExecution.PORTABLE, qr.execution)
        assertEquals(BackendRouteReason.BACKEND_UNAVAILABLE, qr.reason)
    }

    @Test
    fun `the LU still reports natively when its own library resolved`() {
        val adapter = SplitLibraryAdapter(nativeAvailable = true, qrAvailable = false)

        val lu = adapter.route(F64RouteQuery.SparseLu(1))!!

        assertEquals(BackendExecution.NATIVE, lu.execution)
        assertEquals(BackendRouteReason.NATIVE_ROUTE, lu.reason)
    }

    @Test
    fun `a backend whose routines share one library answers both from it`() {
        val adapter = SplitLibraryAdapter(nativeAvailable = true, qrAvailable = true)

        assertEquals(BackendExecution.NATIVE, adapter.route(F64RouteQuery.SparseQr(1))!!.execution)
        assertEquals(BackendExecution.NATIVE, adapter.route(F64RouteQuery.SparseLu(1))!!.execution)
    }
}
