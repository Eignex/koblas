package com.eignex.koblas.internal.operands

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.workspace.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertSame

class StagingTest {

    @Test
    fun `a staged vector holds the entries it addresses and no more`() {
        val original = DoubleArray(64) { it.toDouble() }
        val backing = original.copyOf()
        val views = listOf(
            StridedVector(backing, 0, 64),
            StridedVector(backing, 40, 4),
            StridedVector(backing, 9, 4, 5),
            StridedVector(backing, 33, 4, -8),
        )

        for (x in views) {
            val expected = x.toDoubleArray()
            val workspace = Workspace()

            staged(workspace.arrays, x, aliased = true) { copy ->
                assertEquals(x.size, copy.values.size, "$x staged its buffer rather than its entries")
                assertContentEquals(expected, copy.toDoubleArray(), "$x")
            }

            assertEquals(1, workspace.available(x.size), "$x did not hand its loan back")
            assertContentEquals(original, backing, "$x was written through")
        }
    }

    @Test
    fun `a disjoint vector is passed through without scratch`() {
        val x = StridedVector(DoubleArray(64), 33, 4, -8)
        val workspace = Workspace()

        staged(workspace.arrays, x, aliased = false) { values -> assertSame(x, values) }

        assertEquals(0, workspace.idleLengths())
    }

    /**
     * A matrix-vector product whose input is also its destination, against the oracle's answer for the same
     * numbers held apart.
     */
    @Test
    fun `a staged vector gives the answer the reference computes from unaliased operands`() {
        val rng = Random(20260927)
        val n = 7
        val workspace = Workspace()
        for (transpose in booleanArrayOf(false, true)) {
            val a = randomMatrix(n, n, rng)
            val start = randomVector(n, rng)
            val expected = start.copyOf()
            ReferenceBlas.gemv(0.875, a, start.copyOf(), -0.25, expected, transpose)
            val aliased = start.copyOf()

            blas.withWorkspace(workspace).gemv(0.875, a, aliased, -0.25, aliased, transpose)

            assertClose(expected, aliased, "gemv transpose=$transpose")
        }
    }

    /**
     * Every Level 2 routine that can snapshot an operand takes that snapshot from the workspace.
     *
     * Which overlap is reachable differs by routine, because a matrix's array is as long as all of it while
     * a vector is as long as one side: a product reaches the matrix through a single column, and a routine
     * requiring a square matrix reaches it only at order one. What is asserted is the loan either way, since
     * the answers these produce are the alias suite's subject.
     */
    @Test
    fun `every level two routine stages its alias from the workspace`() {
        val rng = Random(20260932)
        val n = 4

        assertStaged(n, "gemv over the vector it writes") { w ->
            val y = randomVector(n, rng)
            blas.withWorkspace(w).gemv(0.875, randomMatrix(n, n, rng), y, -0.25, y, false)
        }
        assertStaged(n, "gemv over the matrix it writes") { w ->
            val y = randomVector(n, rng)
            blas.withWorkspace(w).gemv(0.875, DenseMatrix.wrap(n, 1, y), doubleArrayOf(2.0), -0.25, y, false)
        }
        assertStaged(n, "symv over the vector it writes") { w ->
            val y = randomVector(n, rng)
            blas.withWorkspace(w).symv(0.875, randomMatrix(n, n, rng), y, -0.25, y, true)
        }
        assertStaged(n, "ger over the vector that is its matrix") { w ->
            val values = randomVector(n, rng)
            blas.withWorkspace(w).ger(0.875, values, doubleArrayOf(2.0), DenseMatrix.wrap(n, 1, values))
        }
        assertStaged(1, "syr over the vector that is its matrix") { w ->
            val values = doubleArrayOf(2.0)
            blas.withWorkspace(w).syr(0.875, DenseVector.wrap(values), one(values), true)
        }
        assertStaged(1, "syr2 over the vector that is its matrix") { w ->
            val values = doubleArrayOf(2.0)
            val other = DenseVector.wrap(doubleArrayOf(3.0))
            blas.withWorkspace(w).syr2(0.875, DenseVector.wrap(values), other, one(values), true)
        }
        assertStaged(1, "trsv over the triangle that is its right-hand side") { w ->
            val values = doubleArrayOf(2.0)
            blas.withWorkspace(w).trsv(one(values), values, lower = true)
        }
        assertStaged(1, "trmv over the triangle that is its right-hand side") { w ->
            val values = doubleArrayOf(2.0)
            blas.withWorkspace(w).trmv(one(values), values, lower = true)
        }
    }

    /** The one-entry matrix over [values], which is the only shape a square routine can alias through. */
    private fun one(values: DoubleArray): DenseMatrix = DenseMatrix.wrap(1, 1, values)

    /** That [call] took exactly one loan, of [length], and returned it. */
    private fun assertStaged(length: Int, what: String, call: (Workspace) -> Unit) {
        val workspace = Workspace()

        call(workspace)

        assertEquals(1, workspace.available(length), "$what did not stage from the workspace")
        assertEquals(1, workspace.idleLengths(), "$what borrowed scratch besides its snapshot")
    }

    private val blas: KoblasEngine get() = BuiltinEngines.scalar
}
