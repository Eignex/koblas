package com.eignex.koblas

import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.sparse.IndexedSparseKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.vendor.RouteKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class VectorRouteTest {
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `an empty run is no work rather than an exact measurement`() {
        val engine = BuiltinEngines.scalar

        for (operation in DenseOperation.entries) {
            val route = engine.routeOf(operation, 0)

            assertEquals(RouteKind.NoWork, route.kind, operation.name)
            assertFalse(route.exactlyMeasurable, operation.name)
        }
        for (operation in SparseOperation.entries) {
            val route = engine.routeOf(operation, 0)

            assertEquals(RouteKind.NoWork, route.kind, operation.name)
            assertFalse(route.exactlyMeasurable, operation.name)
        }
    }

    @Test
    fun `an indexed kernel whose values decide the result is composed rather than exact`() {
        val kernels = SparseKernelAdapter("value-decided", ScalarVectorKernels, ValueDecidedIndexedKernels)

        val route = kernels.routeOf(SparseOperation.IndexedNrm2, 64)

        assertEquals(RouteKind.Composed, route.kind)
        assertFalse(route.exactlyMeasurable)
    }

    @OptIn(KoblasEngineApi::class)
    @Test
    fun `a negative run is rejected in both vector families`() {
        val engine = BuiltinEngines.scalar

        assertFailsWith<IllegalArgumentException> { engine.routeOf(DenseOperation.Dot, -1) }
        assertFailsWith<IllegalArgumentException> { engine.routeOf(SparseOperation.DotDense, -1) }
    }
}

/** Models an indexed norm that may retry with a different kernel. */
private object ValueDecidedIndexedKernels : IndexedSparseKernels by ScalarIndexedSparseKernels {
    override val name: String = "value-decided"

    override fun implementationFor(operation: SparseOperation, count: Int): String? =
        if (operation == SparseOperation.IndexedNrm2) null else name
}
