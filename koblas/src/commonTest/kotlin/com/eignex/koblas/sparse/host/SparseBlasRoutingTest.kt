package com.eignex.koblas.sparse.host

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.test.*

class SparseBlasRoutingTest {

    private open class RecordingAdapter(level2Min: Int, override val nativeAvailable: Boolean = true) :
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

    @Test
    fun `triangular solves report the portable implementation`() {
        val adapter = RecordingAdapter(level2Min = 0)

        val route = adapter.route(F64RouteQuery.SparseTriangularSolve(100, rightHandSides = 4))!!

        assertEquals(BackendExecution.PORTABLE, route.execution)
        assertEquals(BackendRouteReason.UNSUPPORTED_OPERATION, route.reason)
        assertEquals("reference", route.executor)
        assertNull(route.gate)
    }

    @Test
    fun `a sparse adapter can specialize triangular solves independently`() {
        var vectorCalls = 0
        var matrixCalls = 0
        val adapter = object : RecordingAdapter(level2Min = 0) {
            override fun route(query: F64RouteQuery): BackendRoute? =
                if (query is F64RouteQuery.SparseTriangularSolve) {
                    triangularRoute(query, minimumStoredEntries = 2, supported = query.rightHandSides == 1)
                } else {
                    super.route(query)
                }

            override fun trsv(
                a: F64SparseMatrix,
                x: DoubleArray,
                lower: Boolean,
                transpose: Boolean,
                unitDiag: Boolean,
            ) {
                vectorCalls++
                portable.trsv(a, x, lower, transpose, unitDiag)
            }

            @Suppress("LongParameterList")
            override fun trsm(
                a: F64SparseMatrix,
                b: F64DenseMatrix,
                lower: Boolean,
                transpose: Boolean,
                unitDiag: Boolean,
                right: Boolean,
                alpha: Double,
            ) {
                matrixCalls++
                portable.trsm(a, b, lower, transpose, unitDiag, right, alpha)
            }
        }
        val triangle = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))
        val vector = doubleArrayOf(2.0, 4.0)
        val matrix = F64DenseMatrix.diagonal(2)

        adapter.trsv(triangle, vector, lower = true, transpose = false, unitDiag = false)
        adapter.trsm(
            triangle,
            matrix,
            lower = true,
            transpose = false,
            unitDiag = false,
            right = false,
            alpha = 1.0,
        )

        assertEquals(1, vectorCalls)
        assertEquals(1, matrixCalls)
        assertContentEquals(doubleArrayOf(1.0, 1.0), vector)
        assertContentEquals(doubleArrayOf(0.5, 0.0, 0.0, 0.25), matrix.data)
        assertEquals(
            BackendExecution.NATIVE,
            adapter.route(F64RouteQuery.SparseTriangularSolve(2))!!.execution,
        )
        assertEquals(
            BackendRouteReason.UNSUPPORTED_ARGUMENTS,
            adapter.route(F64RouteQuery.SparseTriangularSolve(2, rightHandSides = 2))!!.reason,
        )
    }

    @Test
    fun `native only rejects an unimplemented triangular solve before mutation`() {
        val adapter = RecordingAdapter(level2Min = 0)
        val context = F64ContextBuilder()
            .withBackend(BackendRole.SPARSE_BLAS, adapter)
            .withDispatchPolicy(F64DispatchPolicy.NATIVE_ONLY)
            .resolve()
        val triangle = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 4.0)))
        val vector = doubleArrayOf(2.0, 4.0)

        val failure = assertFailsWith<BackendRouteRejectedException> {
            context.trsv(triangle, vector, lower = true)
        }

        assertIs<F64RouteQuery.SparseTriangularSolve>(failure.route.query)
        assertEquals(BackendRouteReason.UNSUPPORTED_OPERATION, failure.route.reason)
        assertContentEquals(doubleArrayOf(2.0, 4.0), vector)
    }
}
