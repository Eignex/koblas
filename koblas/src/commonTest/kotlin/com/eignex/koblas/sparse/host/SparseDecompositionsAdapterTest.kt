package com.eignex.koblas.sparse.host

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.platformSparseDispatchThresholds
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseQrFactorization
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SparseDecompositionsAdapterTest {

    /** An adapter that accelerates nothing and counts what the gates let through. */
    private class RecordingAdapter(factorizeMin: Int? = null, qrFactorizeMin: Int? = null) :
        F64SparseDecompositionsAdapter(factorizeMin, qrFactorizeMin = qrFactorizeMin) {
        var factors = 0
        var choleskys = 0
        var ldls = 0
        var qrs = 0

        override val name: String get() = "recording"
        override val nativeAvailable: Boolean get() = true

        override fun factorNative(a: F64SparseMatrix): F64SparseFactorization {
            factors++
            return portable.factor(a)
        }

        override fun choleskyNative(a: F64SparseMatrix): F64SparseFactorization {
            choleskys++
            return portable.cholesky(a)
        }

        override fun ldlNative(a: F64SparseMatrix): F64SparseFactorization {
            ldls++
            return portable.ldl(a)
        }

        override fun qrNative(a: F64SparseMatrix): F64SparseQrFactorization {
            qrs++
            return portable.qr(a)
        }
    }

    /** A diagonal of [n] entries, whose stored count is [n] and which every factorization here accepts. */
    private fun diagonal(n: Int) = F64SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 2.0)) })

    @Test
    fun `the QR uses its independently measured gate`() {
        val thresholds = platformSparseDispatchThresholds
        val adapter = RecordingAdapter()

        adapter.qr(diagonal(thresholds.qr - 1))
        adapter.qr(diagonal(thresholds.qr))

        assertEquals(1, adapter.qrs)
    }

    @Test
    fun `the symmetric factorizations gate later than the general one`() {
        val thresholds = platformSparseDispatchThresholds
        assertTrue(
            thresholds.symmetric > thresholds.factorize,
            "the symmetric gate should sit past the general one, got ${thresholds.symmetric} " +
                "against ${thresholds.factorize}",
        )
        val between = (thresholds.factorize + thresholds.symmetric) / 2
        val adapter = RecordingAdapter()

        adapter.factor(diagonal(between))
        adapter.cholesky(diagonal(between))
        adapter.ldl(diagonal(between))

        assertEquals(1, adapter.factors, "the general factorization is past its gate here")
        assertEquals(0, adapter.choleskys, "the Cholesky is not past its own gate here")
        assertEquals(0, adapter.ldls, "the LDL is not past its own gate here")
    }

    @Test
    fun `a backend naming its general gate moves the LU and symmetric factorizations`() {
        val adapter = RecordingAdapter(factorizeMin = 0)

        adapter.factor(diagonal(4))
        adapter.cholesky(diagonal(4))
        adapter.ldl(diagonal(4))
        adapter.qr(diagonal(4))

        assertEquals(1, adapter.factors)
        assertEquals(1, adapter.choleskys, "a caller asking for one gate means it for the symmetric ones too")
        assertEquals(1, adapter.ldls)
        assertEquals(0, adapter.qrs, "the sparse QR retains its independently measured gate")
    }

    @Test
    fun `a backend can override its sparse QR gate independently`() {
        val adapter = RecordingAdapter(factorizeMin = Int.MAX_VALUE, qrFactorizeMin = 0)

        adapter.factor(diagonal(4))
        adapter.qr(diagonal(4))

        assertEquals(0, adapter.factors)
        assertEquals(1, adapter.qrs)
    }

    @Test
    fun `past the last gate all four reach the library`() {
        val adapter = RecordingAdapter()
        val lastGate = maxOf(platformSparseDispatchThresholds.symmetric, platformSparseDispatchThresholds.qr)
        val large = diagonal(lastGate + 1)

        adapter.factor(large)
        adapter.cholesky(large)
        adapter.ldl(large)
        adapter.qr(large)

        assertEquals(1, adapter.factors)
        assertEquals(1, adapter.choleskys)
        assertEquals(1, adapter.ldls)
        assertEquals(1, adapter.qrs)
    }

    @Test
    fun `the sparse LU route reports both sides of its gate`() {
        val adapter = RecordingAdapter(factorizeMin = 8)

        val below = adapter.route(F64RouteQuery.SparseLu(7))!!
        val native = adapter.route(F64RouteQuery.SparseLu(8))!!

        assertEquals(BackendExecution.PORTABLE, below.execution)
        assertEquals(BackendRouteReason.BELOW_THRESHOLD, below.reason)
        assertEquals(DispatchGate(DispatchMetric.STORED_ENTRIES, 7, 8), below.gate)
        assertEquals(BackendExecution.NATIVE, native.execution)
        assertEquals(BackendRouteReason.NATIVE_ROUTE, native.reason)
    }

    @Test
    fun `the sparse QR route reports both sides of its own gate`() {
        val adapter = RecordingAdapter(factorizeMin = 99, qrFactorizeMin = 8)

        val below = adapter.route(F64RouteQuery.SparseQr(7))!!
        val native = adapter.route(F64RouteQuery.SparseQr(8))!!

        assertEquals(BackendExecution.PORTABLE, below.execution)
        assertEquals(BackendRouteReason.BELOW_THRESHOLD, below.reason)
        assertEquals(DispatchGate(DispatchMetric.STORED_ENTRIES, 7, 8), below.gate)
        assertEquals(BackendExecution.NATIVE, native.execution)
        assertEquals(BackendRouteReason.NATIVE_ROUTE, native.reason)
    }
}
