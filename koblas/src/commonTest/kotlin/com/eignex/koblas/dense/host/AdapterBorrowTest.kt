package com.eignex.koblas.dense.host

import com.eignex.koblas.*
import com.eignex.koblas.internal.backend.DispatchThresholds
import kotlin.test.*

/**
 * The host adapters borrow scratch around calls into a library that can raise, and a buffer never handed back
 * is one its pool can neither lend again nor reclaim.
 */
class AdapterBorrowTest {

    /** Calls used only to construct an adapter for routing tests; no numerical entry point should be reached. */
    private class FailingSyrk : CblasCalls {
        override fun dscal(n: Int, alpha: Double, x: DoubleArray, incx: Int) = error("unused")

        override fun daxpy(n: Int, alpha: Double, x: DoubleArray, incx: Int, y: DoubleArray, incy: Int) =
            error("unused")

        @Suppress("LongParameterList")
        override fun dgemv(
            order: Int,
            trans: Int,
            m: Int,
            n: Int,
            alpha: Double,
            a: DoubleArray,
            lda: Int,
            x: DoubleArray,
            incx: Int,
            beta: Double,
            y: DoubleArray,
            incy: Int,
        ) = error("unused")

        @Suppress("LongParameterList")
        override fun dger(
            order: Int,
            m: Int,
            n: Int,
            alpha: Double,
            x: DoubleArray,
            incx: Int,
            y: DoubleArray,
            incy: Int,
            a: DoubleArray,
            lda: Int,
        ) = error("unused")

        @Suppress("LongParameterList")
        override fun dsymv(
            order: Int,
            uplo: Int,
            n: Int,
            alpha: Double,
            a: DoubleArray,
            lda: Int,
            x: DoubleArray,
            incx: Int,
            beta: Double,
            y: DoubleArray,
            incy: Int,
        ) = error("unused")

        @Suppress("LongParameterList")
        override fun dtrsv(
            order: Int,
            uplo: Int,
            trans: Int,
            diag: Int,
            n: Int,
            a: DoubleArray,
            lda: Int,
            x: DoubleArray,
            incx: Int,
        ) = error("unused")

        @Suppress("LongParameterList")
        override fun dtrmv(
            order: Int,
            uplo: Int,
            trans: Int,
            diag: Int,
            n: Int,
            a: DoubleArray,
            lda: Int,
            x: DoubleArray,
            incx: Int,
        ) = error("unused")

        @Suppress("LongParameterList")
        override fun dgemm(
            order: Int,
            transA: Int,
            transB: Int,
            m: Int,
            n: Int,
            k: Int,
            alpha: Double,
            a: DoubleArray,
            lda: Int,
            b: DoubleArray,
            ldb: Int,
            beta: Double,
            c: DoubleArray,
            ldc: Int,
        ) = error("unused")

        @Suppress("LongParameterList")
        override fun dsyrk(
            order: Int,
            uplo: Int,
            trans: Int,
            n: Int,
            k: Int,
            alpha: Double,
            a: DoubleArray,
            lda: Int,
            beta: Double,
            c: DoubleArray,
            ldc: Int,
        ): Unit = error("libopenblas.so.0: cannot open shared object file")

        @Suppress("LongParameterList")
        override fun dsymm(
            order: Int,
            side: Int,
            uplo: Int,
            m: Int,
            n: Int,
            alpha: Double,
            a: DoubleArray,
            lda: Int,
            b: DoubleArray,
            ldb: Int,
            beta: Double,
            c: DoubleArray,
            ldc: Int,
        ) = error("unused")

        @Suppress("LongParameterList")
        override fun dtrsm(
            order: Int,
            side: Int,
            uplo: Int,
            trans: Int,
            diag: Int,
            m: Int,
            n: Int,
            alpha: Double,
            a: DoubleArray,
            lda: Int,
            b: DoubleArray,
            ldb: Int,
        ) = error("unused")

        @Suppress("LongParameterList")
        override fun dtrmm(
            order: Int,
            side: Int,
            uplo: Int,
            trans: Int,
            diag: Int,
            m: Int,
            n: Int,
            alpha: Double,
            a: DoubleArray,
            lda: Int,
            b: DoubleArray,
            ldb: Int,
        ) = error("unused")
    }

    private class FailingAdapter(level2Min: Int = 0, level3Min: Int = 0, private val available: Boolean = true) :
        F64BlasAdapter(
            FailingSyrk(),
            dispatch = DispatchThresholds(0, level2Min, level3Min, 0),
        ) {
        override val name: String get() = "failing-syrk"
        override val isAvailable: Boolean get() = available
    }

    @Test
    fun `dense routes report dimension and work gates`() {
        val adapter = FailingAdapter(level2Min = 16, level3Min = 8)

        val gemv = adapter.route(F64RouteQuery.DenseGemv(15, 100))!!
        val gemm = adapter.route(F64RouteQuery.DenseGemm(64, 2, 4))!!

        assertEquals(BackendRouteReason.BELOW_THRESHOLD, gemv.reason)
        assertEquals(DispatchGate(DispatchMetric.DIMENSION, 15, 16), gemv.gate)
        assertEquals(BackendExecution.NATIVE, gemm.execution)
        assertEquals(DispatchGate(DispatchMetric.LEVEL3_WORK, 512, 512), gemm.gate)
        assertEquals("LP64", gemm.selected.metadata.integerAbi)
    }

    @Test
    fun `dense routes distinguish threshold fallback from unavailable bindings`() {
        val adapter = FailingAdapter(level2Min = 16, available = false)

        val belowThreshold = adapter.route(F64RouteQuery.DenseGemv(15, 100))!!
        val unavailable = adapter.route(F64RouteQuery.DenseGemv(16, 100))!!

        assertEquals(BackendExecution.PORTABLE, belowThreshold.execution)
        assertEquals(BackendRouteReason.BELOW_THRESHOLD, belowThreshold.reason)
        assertEquals(BackendExecution.UNAVAILABLE, unavailable.execution)
        assertEquals(BackendRouteReason.BACKEND_UNAVAILABLE, unavailable.reason)
    }
}
