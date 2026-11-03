package com.eignex.koblas.workspace

import com.eignex.koblas.*
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.engine.*
import com.eignex.koblas.engine.BuiltinEngines
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

class WorkspaceOperationsTest {
    @Test
    fun `binding preserves selected arithmetic and provider metadata`() {
        for (engine in listOf(BuiltinEngines.scalar, koblas)) {
            val bound = engine.withWorkspace(Workspace())

            assertEquals(engine.providers, bound.providers)
            assertSame(engine.panelProvider, bound.panelProvider)
            assertSame(engine.productProvider, bound.productProvider)
            assertSame(engine.triangularProvider, bound.triangularProvider)
            for (length in intArrayOf(1, 4096)) {
                for (operation in EngineImplementation.entries.filter { it.vector }) {
                    assertEquals(
                        engine.implementationFor(operation, length),
                        bound.implementationFor(operation, length),
                    )
                }
            }
        }
    }

    @Test
    fun `rebinding replaces the workspace without stacking adapters`() {
        val engine = BuiltinEngines.scalar
        val first = Workspace()
        val second = Workspace()
        val bound = engine.withWorkspace(first)

        assertSame(bound, engine.withWorkspace(first))
        assertSame(bound, bound.withWorkspace(first))
        assertSame(engine.withWorkspace(second), bound.withWorkspace(second))
        assertSame(engine, bound.withWorkspace(null))
    }

    @Test
    fun `a failed operand read returns all borrowed buffers`() {
        val workspace = Workspace()
        val engine = BuiltinEngines.scalar
        val destination = DenseMatrix.zero(2, 2)
        val before = destination.values.copyOf()
        val broken = object : Matrix by destination {
            override val rows = 2
            override val cols = 2
            override fun get(i: Int, j: Int): Double = error("operand read failed")
        }

        assertFailsWith<IllegalStateException> {
            engine.withWorkspace(workspace).gemm(1.0, broken, false, DenseMatrix.diagonal(2), false, 0.0, destination)
        }

        assertContentEquals(before, destination.values)
        assertTrue(workspace.available(4) > 0)
        val source = randomMatrix(2, 2, Random(74))
        val expected = destination.copyOf()
        ReferenceBlas.gemm(1.0, source, false, source, false, 0.0, expected)
        engine.withWorkspace(workspace).gemm(1.0, source, false, source, false, 0.0, destination)
        assertClose(expected.values, destination.values, "workspace destination")
    }

    @Test
    fun `nested calls keep the outer borrowed arrays intact`() {
        val workspace = Workspace()
        val engine = BuiltinEngines.scalar
        val source = randomMatrix(2, 2, Random(75))
        val inner = source.copyOf()
        val expectedInner = source.copyOf()
        val destination = DenseMatrix.zero(2, 2)
        val identity = DenseMatrix.diagonal(2)
        val expected = destination.copyOf()
        val nested = object : Matrix by source {
            override val rows = source.rows
            override val cols = source.cols
            override fun get(i: Int, j: Int): Double {
                engine.withWorkspace(workspace).gemm(0.75, source, false, inner, false, 0.0, inner)
                return source[i, j]
            }
        }
        repeat(4) {
            ReferenceBlas.gemm(0.75, source, false, expectedInner.copyOf(), false, 0.0, expectedInner)
        }
        ReferenceBlas.gemm(1.0, source, false, identity, false, 0.0, expected)

        engine.withWorkspace(workspace).gemm(1.0, nested, false, identity, false, 0.0, destination)

        assertClose(expected.values, destination.values, "outer workspace destination")
        assertClose(expectedInner.values, inner.values, "nested workspace destination")
    }
}
