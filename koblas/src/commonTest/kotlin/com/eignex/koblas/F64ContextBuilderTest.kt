package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import kotlin.test.*

class F64ContextBuilderTest {

    private class TrackingSparseDecompositions :
        F64SparseDecompositions by F64ReferenceSparseLinearAlgebra,
        F64GeneralSparseLu,
        F64SparseCholesky,
        F64SparseLdl,
        F64SparseQr {
        override val name: String get() = "tracking sparse decompositions"
        var qrCalls: Int = 0

        override fun qr(a: F64SparseMatrix): F64SparseQrFactorization {
            qrCalls++
            return F64ReferenceSparseLinearAlgebra.qr(a)
        }
    }

    private class CountingKernels : F64Kernels by F64PlatformKernels {
        var axpys: Int = 0
        var scales: Int = 0
        override val name: String get() = "counting"

        override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
            axpys++
            F64PlatformKernels.axpy(y, yOff, alpha, x, xOff, len)
        }

        override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
            scales++
            F64PlatformKernels.scale(v, vOff, alpha, len)
        }
    }

    private class RoutedBlas(private val nativeMin: Int) :
        F64Blas by F64ReferenceLinearAlgebra,
        F64RoutingBackend {
        var calls: Int = 0
        override val name: String get() = "routed"
        override val priority: Int get() = 100
        override val isPortable: Boolean get() = false

        override fun route(query: F64RouteQuery): BackendRoute? {
            if (query !is F64RouteQuery.DenseGemv) return null
            val actual = minOf(query.rows, query.cols)
            return BackendRoute(
                query,
                BackendStatus(
                    BackendRole.DENSE_BLAS,
                    name,
                    priority,
                    available = true,
                    portable = false,
                    accelerated = true,
                    BackendMetadata(),
                ),
                if (actual >= nativeMin) BackendExecution.NATIVE else BackendExecution.PORTABLE,
                if (actual >= nativeMin) name else F64ReferenceLinearAlgebra.name,
                if (actual >= nativeMin) BackendRouteReason.NATIVE_ROUTE else BackendRouteReason.BELOW_THRESHOLD,
                DispatchGate(DispatchMetric.DIMENSION, actual.toLong(), nativeMin.toLong()),
            )
        }

        override fun gemv(
            alpha: Double,
            a: F64DenseMatrix,
            x: DoubleArray,
            beta: Double,
            y: DoubleArray,
            transpose: Boolean,
        ) {
            calls++
            F64ReferenceLinearAlgebra.gemv(alpha, a, x, beta, y, transpose)
        }
    }

    @Test
    fun `builders retain independent exact selections without global mutation`() = withCleanBackends {
        val global = koblas
        val base = F64ContextBuilder()
        val routed = RoutedBlas(nativeMin = 0)

        val portable = base.resolve()
        val native = base.withBackend(BackendRole.DENSE_BLAS, routed).resolve()

        assertIs<F64ReferenceBackend>(portable.blas)
        assertSame(routed, native.blas)
        assertNotSame(portable, native)
        assertSame(global, koblas)
    }

    @Test
    fun `portable halves retain their contexts selected dense kernels`() = withCleanBackends {
        val kernels = CountingKernels()
        val context = F64ContextBuilder()
            .withBackend(BackendRole.DENSE_KERNELS, kernels)
            .resolve()

        context.gemv(F64DenseMatrix(1, 1, doubleArrayOf(2.0)), doubleArrayOf(3.0))
        context.sparseBlas.gemv(
            0.0,
            F64SparseMatrix.ofTriplets(1, 1, intArrayOf(), intArrayOf(), doubleArrayOf()),
            doubleArrayOf(0.0),
            2.0,
            doubleArrayOf(1.0),
        )

        assertEquals(1, kernels.axpys)
        assertEquals(1, kernels.scales)
        assertNotSame(kernels, koblas.kernels)
    }

    @Test
    fun `native only rejects a threshold fallback before invoking the backend`() {
        val routed = RoutedBlas(nativeMin = 2)
        val context = F64ContextBuilder()
            .withBackend(BackendRole.DENSE_BLAS, routed)
            .withDispatchPolicy(F64DispatchPolicy.NATIVE_ONLY)
            .resolve()
        val a = F64DenseMatrix(1, 1, doubleArrayOf(2.0))
        val y = doubleArrayOf(7.0)

        val failure = assertFailsWith<BackendRouteRejectedException> {
            context.gemv(1.0, a, doubleArrayOf(3.0), 0.0, y)
        }

        assertEquals(BackendExecution.PORTABLE, failure.route.execution)
        assertEquals(BackendPolicyDecision.REJECT, context.plan(failure.route.query).decision)
        assertEquals(0, routed.calls)
        assertContentEquals(doubleArrayOf(7.0), y)
    }

    @Test
    fun `native only applies the same route to borrowed views`() {
        val routed = RoutedBlas(nativeMin = 2)
        val context = F64ContextBuilder()
            .withBackend(BackendRole.DENSE_BLAS, routed)
            .withDispatchPolicy(F64DispatchPolicy.NATIVE_ONLY)
            .resolve()
        val matrix = F64DenseMatrix(1, 1, doubleArrayOf(2.0)).asView()
        val x = F64DenseVector(doubleArrayOf(3.0)).asView()
        val outputStorage = doubleArrayOf(7.0, 11.0)
        val y = F64StridedVectorView(outputStorage, offset = 0, size = 1)

        assertFailsWith<BackendRouteRejectedException> {
            context.gemv(1.0, matrix, x, 0.0, y)
        }

        assertEquals(0, routed.calls)
        assertContentEquals(doubleArrayOf(7.0, 11.0), outputStorage)
    }

    @Test
    fun `native only executes a known native route`() {
        val routed = RoutedBlas(nativeMin = 2)
        val context = F64ContextBuilder()
            .withBackend(routed)
            .withDispatchPolicy(F64DispatchPolicy.NATIVE_ONLY)
            .resolve()
        val a = F64DenseMatrix(2, 2, doubleArrayOf(1.0, 0.0, 0.0, 1.0))

        val y = context.gemv(a, doubleArrayOf(3.0, 4.0))

        assertContentEquals(doubleArrayOf(3.0, 4.0), y)
        assertEquals(1, routed.calls)
    }

    @Test
    fun `strict routing preserves argument validation precedence`() {
        val context = F64ContextBuilder()
            .withBackend(RoutedBlas(nativeMin = 100))
            .withDispatchPolicy(F64DispatchPolicy.NATIVE_ONLY)
            .resolve()

        assertFailsWith<DimensionMismatch> {
            context.gemv(F64DenseMatrix(1, 2), doubleArrayOf(1.0))
        }
    }

    @Test
    fun `portable only discards external selections`() {
        val context = F64ContextBuilder()
            .withBackend(RoutedBlas(nativeMin = 0))
            .withDispatchPolicy(F64DispatchPolicy.PORTABLE_ONLY)
            .resolve()

        assertIs<F64ReferenceBackend>(context.blas)
        assertEquals(BackendPolicyDecision.EXECUTE, context.plan(F64RouteQuery.DenseGemv(100, 100)).decision)
    }

    @Test
    fun `warn reports a fallback to the context handler`() {
        val warnings = mutableListOf<BackendRoute>()
        val context = F64ContextBuilder()
            .withBackend(RoutedBlas(nativeMin = 4))
            .withFallbackPolicy(F64FallbackPolicy.WARN)
            .onFallback(warnings::add)
            .resolve()

        context.gemv(F64DenseMatrix(1, 1, doubleArrayOf(2.0)), doubleArrayOf(3.0))

        assertEquals(1, warnings.size)
        assertEquals(BackendRouteReason.BELOW_THRESHOLD, warnings.single().reason)
    }

    @Test
    fun `warn requires an explicit handler`() {
        val builder = F64ContextBuilder().withFallbackPolicy(F64FallbackPolicy.WARN)

        assertFailsWith<IllegalArgumentException> { builder.resolve() }
    }

    @Test
    fun `throw rejects an automatic fallback`() {
        val routed = RoutedBlas(nativeMin = 4)
        val context = F64ContextBuilder()
            .withBackend(routed)
            .withFallbackPolicy(F64FallbackPolicy.THROW)
            .resolve()

        assertFailsWith<BackendRouteRejectedException> {
            context.gemv(F64DenseMatrix(1, 1), doubleArrayOf(1.0))
        }
        assertEquals(0, routed.calls)
    }

    @Test
    fun `a role rejects a backend that does not implement it`() {
        val routed = RoutedBlas(nativeMin = 0)

        assertFailsWith<IllegalArgumentException> {
            F64ContextBuilder().withBackend(BackendRole.SPARSE_BLAS, routed)
        }
    }

    @Test
    fun `a complete sparse backend selects QR with the other decomposition roles`() {
        val backend = TrackingSparseDecompositions()
        val context = F64ContextBuilder()
            .withBackend(backend)
            .resolve()
        val matrix = F64SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 1.0, 1 to 1.0)))

        context.qr(matrix).close()

        assertEquals(1, backend.qrCalls)
        assertSame(backend, context.generalSparseLu)
        assertSame(backend, context.sparseCholesky)
        assertSame(backend, context.sparseLdl)
        assertSame(backend, context.sparseQr)
    }
}
