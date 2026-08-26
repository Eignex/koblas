package com.eignex.koblas.sparse.host

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.test.*

class SparseBlasRoutingTest {

    private class RecordingAdapter(level2Min: Int, override val nativeAvailable: Boolean = true) :
        F64SparseBlasAdapter(level2Min) {
        override val name: String get() = "recording"

        override fun gemmNative(
            alpha: Double,
            a: F64SparseMatrix,
            transposeA: Boolean,
            b: F64DenseMatrix,
            beta: Double,
            c: F64DenseMatrix,
        ) = error("not reached by routing diagnostics")
    }

    @Test
    fun `a sparse dense product reports its measured gate`() {
        val adapter = RecordingAdapter(level2Min = 64)

        val below = adapter.route(F64RouteQuery.SparseDenseGemm(63))!!
        val native = adapter.route(F64RouteQuery.SparseDenseGemm(64))!!

        assertEquals(BackendRouteReason.BELOW_THRESHOLD, below.reason)
        assertEquals(DispatchGate(DispatchMetric.STORED_ENTRIES, 63, 64), below.gate)
        assertEquals(BackendExecution.NATIVE, native.execution)
    }

    @Test
    fun `unsupported sparse product arguments report their portable route`() {
        val adapter = RecordingAdapter(level2Min = 0)

        for (query in listOf(
            F64RouteQuery.SparseDenseGemm(100, right = true),
            F64RouteQuery.SparseDenseGemm(100, transposeDense = true),
        )) {
            val route = adapter.route(query)!!
            assertEquals(BackendExecution.PORTABLE, route.execution)
            assertEquals(BackendRouteReason.UNSUPPORTED_ARGUMENTS, route.reason)
            assertEquals("reference", route.executor)
        }
    }

    @Test
    fun `an unavailable sparse binding reports its fallback`() {
        val adapter = RecordingAdapter(level2Min = 0, nativeAvailable = false)

        val route = adapter.route(F64RouteQuery.SparseDenseGemm(100))!!

        assertEquals(BackendExecution.PORTABLE, route.execution)
        assertEquals(BackendRouteReason.BACKEND_UNAVAILABLE, route.reason)
    }
}
