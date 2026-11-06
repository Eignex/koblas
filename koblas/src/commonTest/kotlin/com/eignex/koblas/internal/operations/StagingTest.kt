package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.engine.BuiltinEngines
import com.eignex.koblas.internal.kernels.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class StagingTest {

    /**
     * A matrix-vector product whose input is also its destination, against the oracle's answer for the same
     * numbers held apart.
     */
    @Test
    fun `a staged vector gives the answer the reference computes from unaliased operands`() {
        val rng = Random(20260927)
        val n = 7
        val workspace = Scratch()
        for (transpose in booleanArrayOf(false, true)) {
            val a = randomMatrix(n, n, rng)
            val start = randomVector(n, rng)
            val expected = start.copyOf()
            ReferenceBlas.gemv(0.875, a, start.copyOf(), -0.25, expected, transpose)
            val aliased = start.copyOf()

            blas.gemv(0.875, a, aliased, 0, 1, -0.25, aliased, 0, 1, transpose, scratch = workspace)

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
            blas.gemv(0.875, randomMatrix(n, n, rng), y, 0, 1, -0.25, y, 0, 1, false, scratch = w)
        }
        assertStaged(n, "gemv over the matrix it writes") { w ->
            val y = randomVector(n, rng)
            blas.gemv(0.875, DenseMatrix.wrap(n, 1, y), doubleArrayOf(2.0), 0, 1, -0.25, y, 0, 1, false, scratch = w)
        }
        assertStaged(n, "symv over the vector it writes") { w ->
            val y = randomVector(n, rng)
            blas.symv(0.875, randomMatrix(n, n, rng), y, 0, 1, -0.25, y, 0, 1, true, scratch = w)
        }
        assertStaged(n, "ger over the vector that is its matrix") { w ->
            val values = randomVector(n, rng)
            blas.ger(0.875, values, 0, 1, doubleArrayOf(2.0), 0, 1, DenseMatrix.wrap(n, 1, values), scratch = w)
        }
        assertStaged(1, "syr over the vector that is its matrix") { w ->
            val values = doubleArrayOf(2.0)
            blas.syr(0.875, values, 0, 1, one(values), true, scratch = w)
        }
        assertStaged(1, "syr2 over the vector that is its matrix") { w ->
            val values = doubleArrayOf(2.0)
            val other = DenseVector.wrap(doubleArrayOf(3.0))
            blas.syr2(
                0.875,
                values,
                0,
                1,
                other.values,
                other.offset,
                other.stride,
                one(values),
                true,
                scratch = w,
            )
        }
        assertStaged(1, "trsv over the triangle that is its right-hand side") { w ->
            val values = doubleArrayOf(2.0)
            blas.trsv(one(values), values, 0, 1, lower = true, scratch = w)
        }
        assertStaged(1, "trmv over the triangle that is its right-hand side") { w ->
            val values = doubleArrayOf(2.0)
            blas.trmv(one(values), values, 0, 1, lower = true, scratch = w)
        }
    }

    /** The one-entry matrix over [values], which is the only shape a square routine can alias through. */
    private fun one(values: DoubleArray): DenseMatrix = DenseMatrix.wrap(1, 1, values)

    /** That [call] took exactly one loan, of [length], and returned it. */
    private fun assertStaged(length: Int, what: String, call: (Scratch) -> Unit) {
        val workspace = Scratch()

        call(workspace)

        assertEquals(1, workspace.available(length), "$what did not stage from the workspace")
        assertEquals(1, workspace.idleLengths(), "$what borrowed scratch besides its snapshot")
    }

    private val blas: KoblasEngine get() = BuiltinEngines.scalar
}
