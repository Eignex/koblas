package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * That a workspace handed to a dense routine is used, and that using it changes nothing. The two are
 * separate: a routine that took the parameter and allocated anyway would still compute the right answer, so
 * the loan is checked as the buffers the workspace holds afterwards.
 */
class DenseWorkspaceTest {
    private val blas: DenseBlas get() = testDenseBlas

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

            blas.gemv(0.875, a, aliased, -0.25, aliased, transpose, workspace)

            assertClose(expected, aliased, "gemv transpose=$transpose")
        }
    }

    /**
     * A product whose two operands are both its destination, which is two staging loans of one length at
     * once. A loan that overwrote the other would show up as the wrong answer rather than as a leak.
     */
    @Test
    fun `a product with both operands sharing the destination stages each of them`() {
        val rng = Random(20260928)
        val n = 6
        val workspace = Workspace()
        val start = randomMatrix(n, n, rng)
        val expected = start.copyOf()
        ReferenceBlas.gemm(0.875, start.copyOf(), false, start.copyOf(), true, -0.25, expected)
        val aliased = start.copyOf()

        blas.gemm(0.875, aliased, false, aliased, true, -0.25, aliased, workspace)

        assertClose(expected.values, aliased.values, "gemm with both operands aliased")
        assertEquals(2, workspace.available(n * n), "the two staged operands did not take a loan each")
    }

    // An order inside one diagonal block reaches nothing but the substitution, so the staged triangle is the
    // whole of what the workspace lent and a second length would be scratch the call had no use for.
    @Test
    fun `a triangular solve stages its triangle from the workspace and borrows nothing else`() {
        val rng = Random(20260929)
        val n = 5
        val workspace = Workspace()
        val start = triangle(randomMatrix(n, n, rng), n)
        val expected = start.copyOf()
        ReferenceBlas.trsm(start.copyOf(), expected, lower = true)
        val aliased = start.copyOf()

        blas.trsm(aliased, aliased, lower = true, workspace = workspace)

        assertClose(expected.values, aliased.values, "trsm with its own triangle", tolerance = 1e-9)
        assertEquals(1, workspace.available(n * n), "the triangle was not staged from the workspace")
        assertEquals(1, workspace.idleLengths(), "the solve borrowed scratch besides its staged triangle")
    }

    // The order is past one diagonal block, since below that the call borrows nothing and a reuse check over
    // no loans would pass without checking anything.
    @Test
    fun `a repeated call over one shape reuses the same scratch`() {
        val rng = Random(20260930)
        val order = TRIANGULAR_DIAGONAL_BLOCK + 8
        val workspace = Workspace()
        val triangle = triangle(randomMatrix(order, order, rng), order)
        val b = randomMatrix(order, 4, rng)

        blas.trmm(triangle, b, lower = true, workspace = workspace)
        val afterOne = workspace.idleLengths()
        repeat(3) { blas.trmm(triangle, b, lower = true, workspace = workspace) }

        assertTrue(afterOne > 0, "a triangular multiply over several blocks borrowed nothing")
        assertEquals(afterOne, workspace.idleLengths(), "a repeated call kept asking for new buffers")
    }

    /** A call whose contract stops before the arithmetic takes no loan, because it stages nothing. */
    @Test
    fun `a call with nothing to compute never reaches the workspace`() {
        val rng = Random(20260931)
        val n = 4
        val workspace = Workspace()
        val a = randomMatrix(n, n, rng)
        val c = randomMatrix(n, n, rng)

        blas.gemm(0.0, a, false, a, false, -0.25, c, workspace)
        blas.gemv(0.0, a, DoubleArray(n), -0.25, DoubleArray(n), false, workspace)

        assertEquals(0, workspace.idleLengths(), "a no-work call borrowed scratch it had no use for")
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
            blas.gemv(0.875, randomMatrix(n, n, rng), y, -0.25, y, false, w)
        }
        assertStaged(n, "gemv over the matrix it writes") { w ->
            val y = randomVector(n, rng)
            blas.gemv(0.875, DenseMatrix.wrap(n, 1, y), doubleArrayOf(2.0), -0.25, y, false, w)
        }
        assertStaged(n, "symv over the vector it writes") { w ->
            val y = randomVector(n, rng)
            blas.symv(0.875, randomMatrix(n, n, rng), y, -0.25, y, true, w)
        }
        assertStaged(n, "ger over the vector that is its matrix") { w ->
            val values = randomVector(n, rng)
            blas.ger(0.875, values, doubleArrayOf(2.0), DenseMatrix.wrap(n, 1, values), w)
        }
        assertStaged(1, "syr over the vector that is its matrix") { w ->
            val values = doubleArrayOf(2.0)
            blas.syr(0.875, DenseVector.wrap(values), DenseMatrix.wrap(1, 1, values), true, w)
        }
        assertStaged(1, "syr2 over the vector that is its matrix") { w ->
            val values = doubleArrayOf(2.0)
            blas.syr2(0.875, DenseVector.wrap(values), DenseVector.wrap(doubleArrayOf(3.0)), one(values), true, w)
        }
        assertStaged(1, "trsv over the triangle that is its right-hand side") { w ->
            val values = doubleArrayOf(2.0)
            blas.trsv(one(values), values, lower = true, workspace = w)
        }
        assertStaged(1, "trmv over the triangle that is its right-hand side") { w ->
            val values = doubleArrayOf(2.0)
            blas.trmv(one(values), values, lower = true, workspace = w)
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

    /** A triangle with a dominant diagonal, so a solve over it is well conditioned. */
    private fun triangle(a: DenseMatrix, n: Int): DenseMatrix {
        val t = a.copyOf()
        for (i in 0 until n) t.values[i + i * n] = 3.0 + i % 2
        return t
    }
}
