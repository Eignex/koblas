package com.eignex.koblas

import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.vendor.RouteKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class KoblasEngineTest {
    private val engines get() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()

    @Test
    fun `portable vector routes identify both dense and sparse implementations`() {
        val engine = BuiltinEngines.scalar

        val dense = engine.routeOf(DenseOperation.Dot, 16)
        val sparse = engine.routeOf(SparseOperation.DotDense, 16)

        assertEquals(RouteKind.Direct, dense.kind)
        assertEquals(RouteKind.Direct, sparse.kind)
        assertEquals("scalar", dense.implementation)
        assertEquals("scalar", sparse.implementation)
        assertTrue(dense.exactlyMeasurable)
        assertTrue(sparse.exactlyMeasurable)
    }

    @Test
    fun `empty vector routes describe no work on every engine`() {
        for (engine in engines) {
            for (operation in DenseOperation.entries) {
                val route = engine.routeOf(operation, 0)
                assertEquals(RouteKind.NoWork, route.kind, operation.name)
                assertFalse(route.exactlyMeasurable)
            }
            for (operation in SparseOperation.entries) {
                val route = engine.routeOf(operation, 0)
                assertEquals(RouteKind.NoWork, route.kind, operation.name)
                assertFalse(route.exactlyMeasurable)
            }
        }
    }

    @Test
    fun `vector routes reject negative lengths on every engine`() {
        for (engine in engines) {
            assertFailsWith<IllegalArgumentException> { engine.routeOf(DenseOperation.Dot, -1) }
            assertFailsWith<IllegalArgumentException> { engine.routeOf(SparseOperation.DotDense, -1) }
        }
    }

    @Test
    fun `short and strided dense calls report the portable fallback`() {
        val engine = BuiltinEngines.simd ?: return

        val short = engine.routeOf(DenseOperation.Dot, 1)
        val strided = engine.routeOf(DenseOperation.Dot, 4096, contiguous = false)

        for (route in listOf(short, strided)) {
            assertEquals(RouteKind.Delegated, route.kind)
            assertEquals("scalar", route.implementation)
            assertFalse(route.exactlyMeasurable)
        }
    }

    @Test
    fun `vector norms do not promise an exact simd implementation before seeing values`() {
        val engine = BuiltinEngines.simd ?: return

        val dense = engine.routeOf(DenseOperation.Nrm2, 4096)
        val indexed = engine.routeOf(SparseOperation.IndexedNrm2, 4096)
        val stored = engine.routeOf(SparseOperation.Nrm2, 4096)

        assertEquals(RouteKind.Composed, dense.kind)
        assertEquals(RouteKind.Composed, stored.kind)
        // Some architectures keep indexed reductions scalar regardless of length.
        assertTrue(indexed.kind == RouteKind.Composed || indexed.kind == RouteKind.Delegated)
        assertFalse(indexed.exactlyMeasurable)
    }

    @Test
    fun `one stored entry never reports a vectorized indexed load`() {
        val engine = BuiltinEngines.simd ?: return

        val route = engine.routeOf(SparseOperation.DotDense, 1)

        assertEquals(RouteKind.Delegated, route.kind)
        assertEquals("scalar", route.implementation)
    }

    @Test
    fun `native rotation routes leave the alias dependent implementation unresolved`() {
        if (koblas.vendor == null) return

        val route = koblas.routeOf(DenseOperation.Rot, 4096)

        assertEquals(RouteKind.Composed, route.kind)
        assertFalse(route.exactlyMeasurable)
    }
}
