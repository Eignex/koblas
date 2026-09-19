package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.vendor.RouteKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * What a sparse matrix call reports about itself.
 *
 * The property under test is that the report names the code that runs rather than the engine that was asked
 * for. A whole sparse call is portable CSC scheduling on every engine, so an arm whose Level 1 kernels are
 * Vector API ones must not come out labelled as a vectorised sparse product.
 */
class SparseMatrixRouteTest {

    @Test
    fun `every sparse matrix operation reports the portable scheduling that owns it`() {
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            for (operation in SparseMatrixOperation.entries) {
                val route = engine.matrixRouteOf(operation, entriesPerColumn = 32, denseRun = 512)
                assertEquals("portable-csc", route.scheduling, "${engine.name} $operation")
                assertEquals("portable-csc", engine.sparseImplementation, engine.name)
                assertTrue(
                    route.implementation.startsWith("portable-csc"),
                    "${engine.name} $operation claimed ${route.implementation}",
                )
            }
        }
    }

    @Test
    fun `an operation whose arithmetic is its own traversal names no leaf`() {
        val route = BuiltinEngines.scalar.matrixRouteOf(SparseMatrixOperation.GemmSparse, entriesPerColumn = 8)

        assertNull(route.leaf)
        assertEquals("portable-csc", route.implementation)
        assertEquals(RouteKind.Direct, route.kind)
        assertTrue(route.exactlyMeasurable)
    }

    @Test
    fun `a scattered gemv names the indexed leaf its columns reach`() {
        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar
        val short = engine.matrixRouteOf(SparseMatrixOperation.Gemv, entriesPerColumn = 1)
        val long = engine.matrixRouteOf(SparseMatrixOperation.Gemv, entriesPerColumn = 4096)

        for (route in listOf(short, long)) {
            val leaf = route.leaf
            assertTrue(leaf != null && leaf.endsWith("/axpy"), "gemv reported $leaf")
            assertEquals("portable-csc+$leaf", route.implementation)
        }
        assertEquals(
            engine.sparseKernels.routeOf(SparseOperation.Axpy, 4096).implementation,
            long.leaf?.substringBefore('/'),
            "the sparse gemv leaf must be the one a Level 1 axpy of that width reaches",
        )
    }

    @Test
    fun `a transposed gemv names the ordered scalar dot its contract fixes`() {
        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar

        val route = engine.matrixRouteOf(SparseMatrixOperation.GemvTransposed, entriesPerColumn = 4096)

        assertEquals("scalar/dotDense", route.leaf, "a transposed CSC reduction is ordered by contract")
        assertTrue(route.reason?.contains("ordered scalar dot") == true, route.reason)
    }

    @Test
    fun `a sparse operand on the right names the dense leaf its column updates reach`() {
        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar

        val route = engine.matrixRouteOf(SparseMatrixOperation.GemmDenseRight, entriesPerColumn = 8, denseRun = 4096)

        val leaf = assertNotNull(route.leaf, "a right-side product calls a dense Level 1 kernel")
        assertTrue(leaf.endsWith("/axpy"), "right-side product reported $leaf")
        assertEquals(
            engine.explain(DenseOperation.Axpy, 4096),
            leaf.substringBefore('/'),
            "the leaf must be the one a dense Level 1 axpy of that width reaches",
        )
    }
}
